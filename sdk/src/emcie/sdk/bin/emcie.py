# mypy: disable-error-code=import-untyped

import asyncio
from dataclasses import dataclass
from datetime import datetime
import os
import sys
import time
from typing import Any, Iterable, Literal, NotRequired, Optional, TypedDict, cast
from urllib.parse import urljoin
import click
import click.shell_completion
import click_completion
import requests
import rich
from rich.text import Text
from tabulate import tabulate
from textwrap import wrap
from tqdm import tqdm


class CoherenceCheckFailure(Exception):
    def __init__(self, contradictions: list[dict[str, Any]]) -> None:
        self.contradictions = contradictions


class AgentDTO(TypedDict):
    id: str
    name: str
    description: Optional[str]


class SessionDTO(TypedDict):
    id: str
    end_user_id: str
    title: Optional[str]


class EventDTO(TypedDict):
    id: str
    creation_utc: datetime
    source: Literal["client", "server"]
    kind: str
    offset: int
    correlation_id: str
    data: Any


class TermDTO(TypedDict):
    id: str
    name: str
    description: str
    synonyms: Optional[list[str]]


class GuidelineDTO(TypedDict):
    id: str
    predicate: str
    action: str


class GuidelineConnectionDTO(TypedDict):
    id: str
    source: GuidelineDTO
    target: GuidelineDTO
    kind: Literal["entails", "suggests"]
    indirect: bool


class GuidelineWithConnectionsDTO(TypedDict):
    guideline: GuidelineDTO
    connections: list[GuidelineConnectionDTO]


class FreshnessRulesDTO(TypedDict):
    months: NotRequired[list[int]]
    days_of_month: NotRequired[list[int]]
    days_of_week: NotRequired[
        list[
            Literal[
                "Sunday",
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
            ]
        ]
    ]
    hours: NotRequired[list[int]]
    minutes: NotRequired[list[int]]
    seconds: NotRequired[list[int]]


class ContextVariableDTO(TypedDict):
    id: str
    name: str
    description: NotRequired[str]
    tool_id: NotRequired[str]
    freshness_rules: NotRequired[FreshnessRulesDTO]


class Actions:
    @staticmethod
    def list_agents(ctx: click.Context) -> list[AgentDTO]:
        response = requests.get(urljoin(ctx.obj.server_address, "agents"))
        response.raise_for_status()
        return cast(list[AgentDTO], response.json()["agents"])  # type: ignore

    @staticmethod
    def create_session(
        ctx: click.Context,
        agent_id: str,
        end_user_id: str,
        title: Optional[str] = None,
    ) -> SessionDTO:
        response = requests.post(
            urljoin(ctx.obj.server_address, "/sessions"),
            json={
                "agent_id": agent_id,
                "end_user_id": end_user_id,
                "title": title,
            },
        )

        response.raise_for_status()

        return cast(SessionDTO, response.json()["session"])  # type: ignore

    @staticmethod
    def list_events(ctx: click.Context, session_id: str) -> list[EventDTO]:
        response = requests.get(urljoin(ctx.obj.server_address, f"/sessions/{session_id}/events"))
        response.raise_for_status()
        return cast(list[EventDTO], response.json()["events"])  # type: ignore

    @staticmethod
    def create_event(ctx: click.Context, session_id: str, message: str) -> EventDTO:
        response = requests.post(
            urljoin(ctx.obj.server_address, f"/sessions/{session_id}/events"),
            json={"content": message},
        )

        response.raise_for_status()

        return cast(EventDTO, response.json())  # type: ignore

    @staticmethod
    def create_term(
        ctx: click.Context,
        agent_id: str,
        name: str,
        description: str,
        synonyms: Optional[str],
    ) -> TermDTO:
        response = requests.post(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/terms"),
            json={
                "name": name,
                "description": description,
                **({"synonyms": synonyms.split(",")} if synonyms else {}),
            },
        )

        response.raise_for_status()

        return cast(TermDTO, response.json())  # type: ignore

    @staticmethod
    def remove_term(ctx: click.Context, agent_id: str, name: str) -> None:
        response = requests.delete(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/terms/{name}")
        )
        response.raise_for_status()

    @staticmethod
    def list_terms(ctx: click.Context, agent_id: str) -> list[TermDTO]:
        response = requests.get(urljoin(ctx.obj.server_address, f"/agents/{agent_id}/terms"))
        response.raise_for_status()
        return cast(list[TermDTO], response.json()["terms"])  # type: ignore

    @staticmethod
    def create_guideline(
        ctx: click.Context,
        agent_id: str,
        predicate: str,
        action: str,
        check: bool,
        index: bool,
    ) -> GuidelineWithConnectionsDTO:
        response = requests.post(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/index/evaluations"),
            json={
                "payloads": [
                    {
                        "kind": "guideline",
                        "predicate": predicate,
                        "action": action,
                    }
                ],
                "coherence_check": check,
                "connection_proposition": index,
            },
        )
        response.raise_for_status()
        evaluation_id = response.json()["evaluation_id"]

        with tqdm(
            total=100,
            desc="Evaluating guideline impact",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
        ) as progress_bar:
            while True:
                time.sleep(0.5)
                response = requests.get(
                    urljoin(ctx.obj.server_address, f"/agents/index/evaluations/{evaluation_id}")
                )
                response.raise_for_status()
                evaluation = response.json()

                if evaluation["status"] in ["pending", "running"]:
                    progress_bar.n = int(evaluation["progress"])
                    progress_bar.refresh()

                    continue

                if evaluation["status"] == "completed":
                    invoice = evaluation["invoices"][0]
                    if invoice["approved"]:
                        progress_bar.n = 100
                        progress_bar.refresh()

                        guideline_response = requests.post(
                            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/guidelines/"),
                            json={
                                "invoices": [invoice],
                            },
                        )
                        guideline_response.raise_for_status()

                        return cast(
                            GuidelineWithConnectionsDTO,
                            guideline_response.json()["items"][0],
                        )

                    else:
                        raise CoherenceCheckFailure(
                            contradictions=invoice["data"]["coherence_checks"]
                        )

                elif evaluation["status"] == "failed":
                    raise ValueError(evaluation["error"])

    @staticmethod
    def remove_guideline(ctx: click.Context, agent_id: str, guideline_id: str) -> None:
        response = requests.delete(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/guidelines/{guideline_id}")
        )
        response.raise_for_status()

    @staticmethod
    def get_guideline(
        ctx: click.Context, agent_id: str, guideline_id: str
    ) -> GuidelineWithConnectionsDTO:
        response = requests.get(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/guidelines/{guideline_id}")
        )
        response.raise_for_status()
        return cast(GuidelineWithConnectionsDTO, response.json())  # type: ignore

    @staticmethod
    def list_guidelines(ctx: click.Context, agent_id: str) -> list[GuidelineDTO]:
        response = requests.get(urljoin(ctx.obj.server_address, f"agents/{agent_id}/guidelines"))
        response.raise_for_status()
        return cast(list[GuidelineDTO], response.json()["guidelines"])  # type: ignore

    @staticmethod
    def create_connection(
        ctx: click.Context,
        agent_id: str,
        source_guideline_id: str,
        target_guideline_id: str,
        kind: str,
    ) -> GuidelineWithConnectionsDTO:
        response = requests.patch(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/guidelines/{source_guideline_id}"),
            json={
                "connections": {
                    "add": [
                        {
                            "source": source_guideline_id,
                            "target": target_guideline_id,
                            "kind": kind,
                        }
                    ],
                },
            },
        )
        response.raise_for_status()
        return cast(GuidelineWithConnectionsDTO, response.json())  # type: ignore

    @staticmethod
    def remove_entailment(
        ctx: click.Context,
        agent_id: str,
        source_guideline_id: str,
        target_guideline_id: str,
    ) -> str:
        guideline = requests.get(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/guidelines/{source_guideline_id}")
        )

        connections: list[GuidelineConnectionDTO] = guideline.json()["connections"]

        if connection := next(
            (
                c
                for c in connections
                if target_guideline_id in [c["source"]["id"], c["target"]["id"]]
            ),
            None,
        ):
            response = requests.patch(
                urljoin(
                    ctx.obj.server_address, f"/agents/{agent_id}/guidelines/{source_guideline_id}"
                ),
                json={
                    "connections": {"remove": [target_guideline_id]},
                },
            )
            response.raise_for_status()
            return connection["id"]

        raise ValueError(
            f"An entailment between {source_guideline_id} and {target_guideline_id} was not found"
        )

    @staticmethod
    def list_variables(ctx: click.Context, agent_id: str) -> list[ContextVariableDTO]:
        response = requests.get(urljoin(ctx.obj.server_address, f"/agents/{agent_id}/variables/"))
        response.raise_for_status()
        return cast(list[ContextVariableDTO], response.json()["variables"])

    @staticmethod
    def create_variable(
        ctx: click.Context,
        agent_id: str,
        name: str,
        description: str,
    ) -> ContextVariableDTO:
        response = requests.post(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/variables"),
            json={
                "name": name,
                "description": description,
            },
        )
        response.raise_for_status()
        return cast(ContextVariableDTO, response.json()["variable"])

    @staticmethod
    def _get_variable_by_name(ctx: click.Context, agent_id: str, name: str) -> ContextVariableDTO:
        variables = Actions.list_variables(ctx, agent_id)

        for variable in variables:
            if variable["name"] == name:
                return variable
        raise ValueError(f"Variable with name '{name}' not found")

    @staticmethod
    def remove_variable(ctx: click.Context, agent_id: str, name: str) -> None:
        variable = Actions._get_variable_by_name(ctx, agent_id, name)
        variable_id = variable["id"]

        response = requests.delete(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/variables/{variable_id}")
        )
        response.raise_for_status()


class Interface:
    @staticmethod
    def _write_success(message: str) -> None:
        rich.print(Text(message, style="bold green"))

    @staticmethod
    def _write_error(message: str) -> None:
        rich.print(Text(message, style="bold red"), file=sys.stderr)

    @staticmethod
    def _print_table(data: Iterable[Any], **kwargs: Any) -> None:
        rich.print(
            tabulate(
                data,
                headers="keys",
                tablefmt="rounded_grid",
                **kwargs,
            )
        )

    @staticmethod
    def list_agents(ctx: click.Context) -> None:
        agents = Actions.list_agents(ctx)

        if not agents:
            rich.print("No data available")
            return

        Interface._print_table(agents)

    @staticmethod
    def get_default_agent(ctx: click.Context) -> str:
        agents = Actions.list_agents(ctx)
        assert agents
        return str(agents[0]["id"])

    @staticmethod
    def view_session(ctx: click.Context, session_id: str) -> None:
        events = Actions.list_events(ctx, session_id)

        if not events:
            rich.print("No data available")
            return

        Interface._print_table(
            [
                {
                    "ID": e["id"],
                    "Creation Date": e["creation_utc"],
                    "Correlation ID": e["correlation_id"],
                    "Source": e["source"],
                    "Offset": e["offset"],
                    "Message": e["data"]["message"],
                }
                for e in events
            ],
            maxcolwidths=[None, None, None, None, None, 40],
        )

    @staticmethod
    def create_session(
        ctx: click.Context,
        agent_id: str,
        end_user_id: str,
        title: Optional[str] = None,
    ) -> None:
        session = Actions.create_session(ctx, agent_id, end_user_id, title)
        Interface._write_success(f"Added session (id={session['id']})")
        Interface._print_table([session])

    @staticmethod
    def create_event(ctx: click.Context, session_id: str, message: str) -> None:
        event = Actions.create_event(ctx, session_id, message)
        Interface._write_success(f"Added event (id={event['id']})")
        Interface._print_table([event])

    @staticmethod
    def chat(ctx: click.Context, session_id: str) -> None:
        def print_message(message_event: dict[str, Any]) -> None:
            role = {"client": "User", "server": "Agent"}[message_event["source"]]
            prefix = Text(
                f"{role}:".ljust(6), style="bold " + {"User": "blue", "Agent": "green"}[role]
            )

            message = wrap(
                message_event["data"]["message"], subsequent_indent=" " * (1 + len(prefix))
            )

            rich.print(prefix, os.linesep.join(message))

        rich.print(Text("Press CTRL+C at any time to quit\n", style="bold"))

        response = requests.get(urljoin(ctx.obj.server_address, f"/sessions/{session_id}/events"))
        response.raise_for_status()

        message_events = [e for e in response.json()["events"] if e["kind"] == "message"]

        max_number_of_history_events_to_show = 5

        if len(message_events) > max_number_of_history_events_to_show:
            rich.print(
                f"(skipping {len(message_events) - max_number_of_history_events_to_show} "
                "event(s) in history...)\n",
                flush=True,
            )
            message_events = message_events[-max_number_of_history_events_to_show:]

        for m in message_events:
            print_message(m)

        last_known_offset = message_events[-1]["offset"] if message_events else -1

        while True:
            try:
                rich.print(Text("User:  ", style="bold blue"), end="")
                new_message = input()

                response = requests.post(
                    urljoin(
                        ctx.obj.server_address,
                        f"/sessions/{session_id}/events",
                    ),
                    json={"content": new_message},
                )
                response.raise_for_status()
                new_event = response.json()

                last_known_offset = new_event["event_offset"]

                while True:
                    response = requests.get(
                        urljoin(
                            ctx.obj.server_address,
                            f"/sessions/{session_id}/events"
                            f"?min_offset={1 + last_known_offset}&wait=true",
                        )
                    )

                    if response.status_code == 504:
                        # Timeout occurred; try again
                        continue

                    events = response.json()["events"]
                    if not events:
                        continue

                    last_known_offset = events[-1]["offset"]

                    message_events = [e for e in events if e["kind"] == "message"]
                    if message_events:
                        for m in message_events:
                            print_message(m)
                        break

            except KeyboardInterrupt:
                rich.print("\nQuitting...", flush=True)
                return

    @staticmethod
    def create_term(
        ctx: click.Context,
        agent_id: str,
        name: str,
        description: str,
        synonyms: Optional[str],
    ) -> None:
        term = Actions.create_term(ctx, agent_id, name, description, synonyms)
        Interface._write_success(f"Added term (id={term['id']})")
        Interface._print_table([term])

    @staticmethod
    def remove_term(ctx: click.Context, agent_id: str, name: str) -> None:
        Actions.remove_term(ctx, agent_id, name)
        Interface._write_success(f"Removed term '{name}'")

    @staticmethod
    def list_terms(ctx: click.Context, agent_id: str) -> None:
        terms = Actions.list_terms(ctx, agent_id)

        if not terms:
            rich.print("No data available")
            return

        Interface._print_table(terms)

    @staticmethod
    def _render_guidelines(guidelines: list[GuidelineDTO]) -> None:
        guideline_items = [
            {
                "ID": guideline["id"],
                "Predicate": guideline["predicate"],
                "Action": guideline["action"],
            }
            for guideline in guidelines
        ]

        Interface._print_table(guideline_items, maxcolwidths=[None, 40, 40])

    @staticmethod
    def _render_guideline_entailments(
        guideline: GuidelineDTO,
        connections: list[GuidelineConnectionDTO],
        include_indirect: bool,
    ) -> None:
        def to_direct_entailment_item(conn: GuidelineConnectionDTO) -> dict[str, str]:
            peer = conn["target"] if conn["source"]["id"] == guideline["id"] else conn["source"]

            return {
                "Connection ID": conn["id"],
                "Entailment": "Strict" if conn["kind"] == "entails" else "Suggestive",
                "Role": "Source" if conn["source"]["id"] == guideline["id"] else "Target",
                "Peer Role": "Target" if conn["source"]["id"] == guideline["id"] else "Source",
                "Peer ID": peer["id"],
                "Peer Predicate": peer["predicate"],
                "Peer Action": peer["action"],
            }

        def to_indirect_entailment_item(conn: GuidelineConnectionDTO) -> dict[str, str]:
            return {
                "Connection ID": conn["id"],
                "Entailment": "Strict" if conn["kind"] == "entails" else "Suggestive",
                "Source ID": conn["source"]["id"],
                "Source Predicate": conn["source"]["predicate"],
                "Source Action": conn["source"]["action"],
                "Target ID": conn["target"]["id"],
                "Target Predicate": conn["target"]["predicate"],
                "Target Action": conn["target"]["action"],
            }

        if connections:
            direct = [c for c in connections if not c["indirect"]]
            indirect = [c for c in connections if c["indirect"]]

            if direct:
                rich.print("\nDirect Entailments:")
                Interface._print_table(map(lambda c: to_direct_entailment_item(c), direct))

            if indirect and include_indirect:
                rich.print("\nIndirect Entailments:")
                Interface._print_table(map(lambda c: to_indirect_entailment_item(c), indirect))

    @staticmethod
    def create_guideline(
        ctx: click.Context,
        agent_id: str,
        predicate: str,
        action: str,
        check: bool,
        index: bool,
    ) -> None:
        try:
            guideline_with_connections = Actions.create_guideline(
                ctx,
                agent_id,
                predicate,
                action,
                check,
                index,
            )

            guideline = guideline_with_connections["guideline"]
            Interface._write_success(f"Added guideline (id={guideline['id']})")

            Interface._render_guideline_entailments(
                guideline_with_connections["guideline"],
                guideline_with_connections["connections"],
                include_indirect=False,
            )

        except CoherenceCheckFailure as e:
            contradictions = e.contradictions
            Interface._write_error("Failed to add guideline")
            rich.print("Detected incoherence with other guidelines:")
            Interface._print_table(
                contradictions,
                maxcolwidths=[20, 20, 20, 40, 10],
            )
            rich.print(
                Text("\nTo force-add despite these errors, re-run with --no-check", style="bold")
            )
        except Exception as e:
            Interface._write_error(f"error: {type(e).__name__}: {e}")

    @staticmethod
    def remove_guideline(ctx: click.Context, agent_id: str, guideline_id: str) -> None:
        try:
            Actions.remove_guideline(ctx, agent_id, guideline_id)
            Interface._write_success(f"Removed guideline (id={guideline_id})")
        except Exception as e:
            Interface._write_error(f"error: {type(e).__name__}: {e}")

    @staticmethod
    def view_guideline(ctx: click.Context, agent_id: str, guideline_id: str) -> None:
        try:
            guideline_with_connections = Actions.get_guideline(ctx, agent_id, guideline_id)

            Interface._render_guidelines([guideline_with_connections["guideline"]])

            Interface._render_guideline_entailments(
                guideline_with_connections["guideline"],
                guideline_with_connections["connections"],
                include_indirect=True,
            )
        except Exception as e:
            Interface._write_error(f"error: {type(e).__name__}: {e}")

    @staticmethod
    def list_guidelines(ctx: click.Context, agent_id: str) -> None:
        try:
            guidelines = Actions.list_guidelines(ctx, agent_id)

            if not guidelines:
                rich.print("No data available")
                return

            Interface._render_guidelines(guidelines)

        except Exception as e:
            Interface._write_error(f"error: {type(e).__name__}: {e}")

    @staticmethod
    def create_entailment(
        ctx: click.Context,
        agent_id: str,
        source_guideline_id: str,
        target_guideline_id: str,
        kind: str,
    ) -> None:
        try:
            connection = Actions.create_connection(
                ctx,
                agent_id,
                source_guideline_id,
                target_guideline_id,
                kind,
            )
            Interface._write_success(f"Added connection (id={connection["connections"][0]['id']})")
            Interface._print_table([connection])
        except Exception as e:
            Interface._write_error(f"error: {type(e).__name__}: {e}")

    @staticmethod
    def remove_entailment(
        ctx: click.Context,
        agent_id: str,
        source_guideline_id: str,
        target_guideline_id: str,
    ) -> None:
        try:
            connection_id = Actions.remove_entailment(
                ctx,
                agent_id,
                source_guideline_id,
                target_guideline_id,
            )
            Interface._write_success(f"Removed entailment (id={connection_id})")
        except Exception as e:
            Interface._write_error(f"error: {type(e).__name__}: {e}")

    @staticmethod
    def _render_freshness_rules(freshness_rules: FreshnessRulesDTO) -> str:
        if freshness_rules is None:
            return ""
        parts = []
        if freshness_rules.get("months"):
            months = ", ".join(str(m) for m in freshness_rules["months"])
            parts.append(f"Months: {months}")
        if freshness_rules.get("days_of_month"):
            days_of_month = ", ".join(str(d) for d in freshness_rules["days_of_month"])
            parts.append(f"Days of Month: {days_of_month}")
        if freshness_rules.get("days_of_week"):
            days_of_week = ", ".join(freshness_rules["days_of_week"])
            parts.append(f"Days of Week: {days_of_week}")
        if freshness_rules.get("hours"):
            hours = ", ".join(str(h) for h in freshness_rules["hours"])
            parts.append(f"Hours: {hours}")
        if freshness_rules.get("minutes"):
            minutes = ", ".join(str(m) for m in freshness_rules["minutes"])
            parts.append(f"Minutes: {minutes}")
        if freshness_rules.get("seconds"):
            seconds = ", ".join(str(s) for s in freshness_rules["seconds"])
            parts.append(f"Seconds: {seconds}")
        if not parts:
            return "None"
        return "; ".join(parts)

    @staticmethod
    def list_variables(ctx: click.Context, agent_id: str) -> None:
        variables = Actions.list_variables(ctx, agent_id)
        if not variables:
            rich.print("No variables found")
            return

        variable_items = [
            {
                "ID": variable["id"],
                "Name": variable["name"],
                "Description": variable["description"] or "",
                "Tool ID": variable["tool_id"] or "",
                "Freshness Rules": Interface._render_freshness_rules(variable["freshness_rules"]),
            }
            for variable in variables
        ]

        Interface._print_table(variable_items)

    @staticmethod
    def create_variable(
        ctx: click.Context,
        agent_id: str,
        name: str,
        description: str,
    ) -> None:
        variable = Actions.create_variable(ctx, agent_id, name, description)

        Interface._write_success(f"Added variable (id={variable['id']})")
        Interface._print_table(
            [
                {
                    "ID": variable["id"],
                    "Name": variable["name"],
                    "Description": variable["description"] or "",
                }
            ],
        )

    @staticmethod
    def remove_variable(ctx: click.Context, agent_id: str, name: str) -> None:
        try:
            Actions.remove_variable(ctx, agent_id, name)
            Interface._write_success(f"Removed variable '{name}'")
        except Exception as e:
            Interface._write_error(f"error: {type(e).__name__}: {e}")


async def async_main() -> None:
    click_completion.init()

    @dataclass(frozen=True)
    class Config:
        server_address: str

    @click.group
    @click.option(
        "-s", "--server", type=str, help="Server address", metavar="ADDRESS[:PORT]", required=True
    )
    @click.pass_context
    def cli(ctx: click.Context, server: str) -> None:
        if not ctx.obj:
            ctx.obj = Config(server_address=server)

    @cli.command(help="Generate shell completion code")
    @click.option("-s", "--shell", type=str, help="Shell program (bash, zsh, etc.)", required=True)
    def complete(shell: str) -> None:
        click.echo(click_completion.get_code(shell))

    @cli.group(help="Manage agents")
    def agent() -> None:
        pass

    @agent.command("list", help="List agents")
    @click.pass_context
    def agent_list(ctx: click.Context) -> None:
        Interface.list_agents(ctx)

    @agent.command(
        "chat",
        help="Jump into a chat with an agent\n\n"
        "If AGENT_ID is omitted, the default agent will be selected.",
    )
    @click.argument("agent_id", required=False)
    @click.pass_context
    def agent_chat(ctx: click.Context, agent_id: Optional[str]) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id
        session = Actions.create_session(ctx, agent_id=agent_id, end_user_id="<unused>")
        Interface.chat(ctx, session["id"])

    @cli.group(help="Manage sessions")
    def session() -> None:
        pass

    @session.command("new", help="Create a new session")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.option("-u", "--end-user-id", type=str, help="End User ID", metavar="ID", required=True)
    @click.option("-t", "--title", type=str, help="Session Title", metavar="TITLE", required=False)
    @click.pass_context
    def session_new(
        ctx: click.Context,
        agent_id: str,
        end_user_id: str,
        title: Optional[str],
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.create_session(ctx, agent_id, end_user_id, title)

    @session.command("view", help="View session content")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.argument("session_id")
    @click.pass_context
    def session_view(ctx: click.Context, agent_id: str, session_id: str) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.view_session(ctx, session_id)

    @session.command("post", help="Post user message to session")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.argument("session_id")
    @click.argument("message")
    @click.pass_context
    def session_post(ctx: click.Context, agent_id: str, session_id: str, message: str) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.create_event(ctx, session_id, message)

    @session.command("chat", help="Enter chat mode within the session")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.argument("session_id")
    @click.pass_context
    def session_chat(ctx: click.Context, agent_id: str, session_id: str) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.chat(ctx, session_id)

    @cli.group(help="Manage an agent's glossary")
    def glossary() -> None:
        pass

    @glossary.command("add", help="Add a new term to the glossary")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.argument("name", type=str)
    @click.argument("description", type=str)
    @click.option(
        "-s", "--synonyms", type=str, help="Comma-separated list of synonyms", required=False
    )
    @click.pass_context
    def glossary_add(
        ctx: click.Context,
        agent_id: str,
        name: str,
        description: str,
        synonyms: Optional[str],
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id
        Interface.create_term(ctx, agent_id, name, description, synonyms)

    @glossary.command("remove", help="Remove a term from the glossary")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.argument("name", type=str)
    @click.pass_context
    def glossary_remove(
        ctx: click.Context,
        agent_id: str,
        name: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id
        Interface.remove_term(ctx, agent_id, name)

    @glossary.command("list", help="List all terms in the glossary")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.pass_context
    def glossary_list(
        ctx: click.Context,
        agent_id: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id
        Interface.list_terms(ctx, agent_id)

    @cli.group(help="Manage an agent's guidelines")
    def guideline() -> None:
        pass

    @guideline.command("add", help="Add a new guideline")
    @click.option(
        "--check/--no-check",
        type=bool,
        show_default=True,
        default=True,
        help="Check for contradictions between existing guidelines",
    )
    @click.option(
        "--index/--no-index",
        type=bool,
        show_default=True,
        default=True,
        help="Determine if guideline connections should be indexed",
    )
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.argument("predicate", type=str)
    @click.argument("action", type=str)
    @click.pass_context
    def guideline_add(
        ctx: click.Context,
        agent_id: str,
        predicate: str,
        action: str,
        check: bool,
        index: bool,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.create_guideline(
            ctx=ctx,
            agent_id=agent_id,
            predicate=predicate,
            action=action,
            check=check,
            index=index,
        )

    @guideline.command("remove", help="Remove a guideline")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.argument("guideline_id", type=str)
    @click.pass_context
    def guideline_remove(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.remove_guideline(
            ctx=ctx,
            agent_id=agent_id,
            guideline_id=guideline_id,
        )

    @guideline.command("view", help="View a guideline and its connections")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.argument("guideline_id", type=str)
    @click.pass_context
    def guideline_view(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.view_guideline(
            ctx=ctx,
            agent_id=agent_id,
            guideline_id=guideline_id,
        )

    @guideline.command("list", help="List all guidelines for an agent")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.pass_context
    def guideline_list(
        ctx: click.Context,
        agent_id: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.list_guidelines(
            ctx=ctx,
            agent_id=agent_id,
        )

    @guideline.command("entail", help="Create an entailment between two guidelines")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.option(
        "--suggestive/-s",
        is_flag=True,
        show_default=True,
        default=False,
        help="Make the entailment suggestive rather than definite",
    )
    @click.argument("source_guideline_id", type=str)
    @click.argument("target_guideline_id", type=str)
    @click.pass_context
    def guideline_entail(
        ctx: click.Context,
        agent_id: str,
        suggestive: bool,
        source_guideline_id: str,
        target_guideline_id: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.create_entailment(
            ctx=ctx,
            agent_id=agent_id,
            source_guideline_id=source_guideline_id,
            target_guideline_id=target_guideline_id,
            kind="suggests" if suggestive else "entails",
        )

    @guideline.command("disentail", help="Remove an entailment between two guidelines")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.argument("source_guideline_id", type=str)
    @click.argument("target_guideline_id", type=str)
    @click.pass_context
    def guideline_disentail(
        ctx: click.Context,
        agent_id: str,
        source_guideline_id: str,
        target_guideline_id: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.remove_entailment(
            ctx=ctx,
            agent_id=agent_id,
            source_guideline_id=source_guideline_id,
            target_guideline_id=target_guideline_id,
        )

    @cli.group(help="Manage an agent's variables")
    def variable() -> None:
        pass

    @variable.command("list", help="List all variables for an agent")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.pass_context
    def variable_list(
        ctx: click.Context,
        agent_id: Optional[str],
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.list_variables(
            ctx=ctx,
            agent_id=agent_id,
        )

    @variable.command("add", help="Add a new variable to an agent")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.option("-d", "--description", type=str, help="Variable description", required=False)
    @click.argument("name", type=str)
    @click.pass_context
    def variable_add(
        ctx: click.Context,
        agent_id: Optional[str],
        name: str,
        description: Optional[str],
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.create_variable(
            ctx=ctx,
            agent_id=agent_id,
            name=name,
            description=description or "",
        )

    @variable.command("remove", help="Remove a variable from an agent")
    @click.option("-a", "--agent-id", type=str, help="Agent ID", metavar="ID", required=False)
    @click.argument("name", type=str)
    @click.pass_context
    def variable_remove(
        ctx: click.Context,
        agent_id: Optional[str],
        name: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.remove_variable(
            ctx=ctx,
            agent_id=agent_id,
            name=name,
        )

    cli()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
