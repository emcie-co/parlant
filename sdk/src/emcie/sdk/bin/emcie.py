# mypy: disable-error-code=import-untyped

import asyncio
from dataclasses import dataclass
from datetime import datetime
import os
import sys
import time
from typing import Any, Iterable, Literal, NotRequired, Optional, TypedDict, Union, cast
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

INDENT = "  "


class CoherenceCheckFailure(Exception):
    def __init__(self, contradictions: list[dict[str, Any]]) -> None:
        self.contradictions = contradictions


class AgentDTO(TypedDict):
    id: str
    name: str
    description: Optional[str]
    creation_utc: str
    max_engine_iterations: int


class SessionDTO(TypedDict):
    id: str
    end_user_id: str
    title: Optional[str]
    creation_utc: str


class EventDTO(TypedDict):
    id: str
    creation_utc: str
    source: Literal["client", "server"]
    kind: str
    offset: int
    correlation_id: str
    data: Any


class TermDTO(TypedDict):
    id: str
    name: str
    description: str
    synonyms: list[str]


class GuidelineDTO(TypedDict):
    id: str
    predicate: str
    action: str


class ToolIdDTO(TypedDict):
    service_name: str
    tool_name: str


class GuidelineConnectionDTO(TypedDict):
    id: str
    source: GuidelineDTO
    target: GuidelineDTO
    kind: Literal["entails", "suggests"]
    indirect: bool


class GuidelineToolAssociationDTO(TypedDict):
    id: str
    guideline_id: str
    tool_id: ToolIdDTO


class GuidelineWithConnectionsAndToolAssociationsDTO(TypedDict):
    guideline: GuidelineDTO
    connections: list[GuidelineConnectionDTO]
    tool_associations: list[GuidelineToolAssociationDTO]


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
    tool_id: NotRequired[ToolIdDTO]
    freshness_rules: NotRequired[FreshnessRulesDTO]


class ContextVariableValueDTO(TypedDict):
    id: str
    last_modified: str
    data: Any


class GuidelinePropositionDTO(TypedDict):
    guideline_id: str
    predicate: str
    action: str
    score: int
    rationale: str


class ToolResultDTO(TypedDict):
    data: Any
    metadata: dict[str, Any]


class ToolCallDTO(TypedDict):
    tool_name: str
    arguments: dict[str, Any]
    result: ToolResultDTO


class InspectedContextVariableDTO(TypedDict):
    id: str
    name: str
    description: str
    key: str
    value: str


class ReadContextVariableResponse(TypedDict):
    context_variable: ContextVariableDTO
    key_value_pairs: dict[str, ContextVariableValueDTO]


class PreparationIterationDTO(TypedDict):
    guideline_propositions: list[GuidelinePropositionDTO]
    tool_calls: list[ToolCallDTO]
    terms: list[TermDTO]
    context_variables: list[InspectedContextVariableDTO]


class ReadInteractionResponse(TypedDict):
    session_id: str
    correlation_id: str
    preparation_iterations: list[PreparationIterationDTO]


class ToolParameterDTO(TypedDict):
    type: str
    description: NotRequired[str]
    enum: NotRequired[list[Union[str, int, float, bool]]]


class ToolDTO(TypedDict):
    creation_utc: datetime
    name: str
    description: str
    parameters: dict[str, ToolParameterDTO]
    required: list[str]
    consequential: bool


class ServiceDTO(TypedDict):
    name: str
    kind: str
    url: str
    tools: Optional[list[ToolDTO]]


def format_datetime(datetime_str: str) -> str:
    return datetime.fromisoformat(datetime_str).strftime("%Y-%m-%d %I:%M:%S %p %Z")


_EXIT_STATUS = 0


def get_exit_status() -> int:
    return _EXIT_STATUS


def set_exit_status(status: int) -> None:
    global _EXIT_STATUS
    _EXIT_STATUS = status


class Actions:
    @staticmethod
    def create_agent(
        ctx: click.Context,
        name: str,
        description: Optional[str],
        max_engine_iterations: Optional[int],
    ) -> AgentDTO:
        response = requests.post(
            urljoin(ctx.obj.server_address, "/agents"),
            json={
                "name": name,
                "description": description,
                "max_engine_iterations": max_engine_iterations,
            },
        )

        response.raise_for_status()

        return cast(AgentDTO, response.json()["agent"])

    @staticmethod
    def read_agent(ctx: click.Context, agent_id: str) -> AgentDTO:
        response = requests.get(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}"),
        )

        response.raise_for_status()

        return cast(AgentDTO, response.json())

    @staticmethod
    def list_agents(ctx: click.Context) -> list[AgentDTO]:
        response = requests.get(urljoin(ctx.obj.server_address, "agents"))

        response.raise_for_status()

        return cast(list[AgentDTO], response.json()["agents"])  # type: ignore

    @staticmethod
    def patch_agent(
        ctx: click.Context,
        agent_id: str,
        description: Optional[str] = None,
        max_engine_iterations: Optional[int] = None,
    ) -> None:
        patch_data: dict[str, Any] = {}
        if description is not None:
            patch_data["description"] = description
        if max_engine_iterations is not None:
            patch_data["max_engine_iterations"] = max_engine_iterations

        response = requests.patch(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}"),
            json=patch_data,
        )

        response.raise_for_status()

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

        return cast(SessionDTO, response.json()["session"])

    @staticmethod
    def list_sessions(
        ctx: click.Context,
        agent_id: Optional[str],
        end_user_id: Optional[str],
    ) -> list[SessionDTO]:
        response = requests.get(
            urljoin(ctx.obj.server_address, "/sessions"),
            params={
                "agent_id": agent_id,
                "end_user_id": end_user_id,
            },
        )

        response.raise_for_status()

        return cast(list[SessionDTO], response.json()["sessions"])

    @staticmethod
    def inspect_interaction(
        ctx: click.Context,
        session_id: str,
        correlation_id: str,
    ) -> ReadInteractionResponse:
        response = requests.get(
            urljoin(ctx.obj.server_address, f"/sessions/{session_id}/interactions/{correlation_id}")
        )

        response.raise_for_status()

        return cast(ReadInteractionResponse, response.json())

    @staticmethod
    def list_events(
        ctx: click.Context,
        session_id: str,
    ) -> list[EventDTO]:
        response = requests.get(urljoin(ctx.obj.server_address, f"/sessions/{session_id}/events"))

        response.raise_for_status()

        return cast(list[EventDTO], response.json()["events"])

    @staticmethod
    def create_event(
        ctx: click.Context,
        session_id: str,
        message: str,
    ) -> EventDTO:
        response = requests.post(
            urljoin(ctx.obj.server_address, f"/sessions/{session_id}/events"),
            json={"content": message},
        )

        response.raise_for_status()

        return cast(EventDTO, response.json()["event"])

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

        return cast(TermDTO, response.json()["term"])

    @staticmethod
    def remove_term(
        ctx: click.Context,
        agent_id: str,
        name: str,
    ) -> None:
        response = requests.delete(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/terms/{name}")
        )

        response.raise_for_status()

    @staticmethod
    def list_terms(
        ctx: click.Context,
        agent_id: str,
    ) -> list[TermDTO]:
        response = requests.get(urljoin(ctx.obj.server_address, f"/agents/{agent_id}/terms"))

        response.raise_for_status()

        return cast(list[TermDTO], response.json()["terms"])

    @staticmethod
    def create_guideline(
        ctx: click.Context,
        agent_id: str,
        predicate: str,
        action: str,
        check: bool,
        index: bool,
    ) -> GuidelineWithConnectionsAndToolAssociationsDTO:
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
                    urljoin(
                        ctx.obj.server_address,
                        f"/agents/index/evaluations/{evaluation_id}",
                    )
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
                            urljoin(
                                ctx.obj.server_address,
                                f"/agents/{agent_id}/guidelines/",
                            ),
                            json={
                                "invoices": [invoice],
                            },
                        )

                        guideline_response.raise_for_status()

                        return cast(
                            GuidelineWithConnectionsAndToolAssociationsDTO,
                            guideline_response.json()["items"][0],
                        )

                    else:
                        raise CoherenceCheckFailure(
                            contradictions=invoice["data"]["coherence_checks"]
                        )

                elif evaluation["status"] == "failed":
                    raise ValueError(evaluation["error"])

    @staticmethod
    def remove_guideline(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
    ) -> None:
        response = requests.delete(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/guidelines/{guideline_id}")
        )

        response.raise_for_status()

    @staticmethod
    def read_guideline(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
    ) -> GuidelineWithConnectionsAndToolAssociationsDTO:
        response = requests.get(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/guidelines/{guideline_id}")
        )

        response.raise_for_status()

        return cast(GuidelineWithConnectionsAndToolAssociationsDTO, response.json())

    @staticmethod
    def list_guidelines(
        ctx: click.Context,
        agent_id: str,
    ) -> list[GuidelineDTO]:
        response = requests.get(urljoin(ctx.obj.server_address, f"agents/{agent_id}/guidelines"))

        response.raise_for_status()

        return cast(list[GuidelineDTO], response.json()["guidelines"])

    @staticmethod
    def add_entailment(
        ctx: click.Context,
        agent_id: str,
        source_guideline_id: str,
        target_guideline_id: str,
        kind: str,
    ) -> GuidelineWithConnectionsAndToolAssociationsDTO:
        response = requests.patch(
            urljoin(
                ctx.obj.server_address,
                f"/agents/{agent_id}/guidelines/{source_guideline_id}",
            ),
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

        return cast(GuidelineWithConnectionsAndToolAssociationsDTO, response.json())

    @staticmethod
    def remove_entailment(
        ctx: click.Context,
        agent_id: str,
        source_guideline_id: str,
        target_guideline_id: str,
    ) -> str:
        guideline_response = requests.get(
            urljoin(
                ctx.obj.server_address,
                f"/agents/{agent_id}/guidelines/{source_guideline_id}",
            )
        )

        guideline_response.raise_for_status()

        connections: list[GuidelineConnectionDTO] = guideline_response.json()["connections"]

        if connection := next(
            (
                c
                for c in connections
                if target_guideline_id in [c["source"]["id"], c["target"]["id"]]
            ),
            None,
        ):
            connection_response = requests.patch(
                urljoin(
                    ctx.obj.server_address,
                    f"/agents/{agent_id}/guidelines/{source_guideline_id}",
                ),
                json={
                    "connections": {"remove": [target_guideline_id]},
                },
            )

            connection_response.raise_for_status()

            return connection["id"]

        raise ValueError(
            f"An entailment between {source_guideline_id} and {target_guideline_id} was not found"
        )

    @staticmethod
    def add_association(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> GuidelineWithConnectionsAndToolAssociationsDTO:
        response = requests.patch(
            urljoin(
                ctx.obj.server_address,
                f"/agents/{agent_id}/guidelines/{guideline_id}",
            ),
            json={
                "tool_associations": {
                    "add": [
                        {
                            "service_name": service_name,
                            "tool_name": tool_name,
                        }
                    ]
                }
            },
        )

        response.raise_for_status()

        return cast(GuidelineWithConnectionsAndToolAssociationsDTO, response.json())

    @staticmethod
    def remove_association(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> str:
        guideline_response = requests.get(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/guidelines/{guideline_id}")
        )

        guideline_response.raise_for_status()

        associations: list[GuidelineToolAssociationDTO] = guideline_response.json()[
            "tool_associations"
        ]

        if association := next(
            (
                a
                for a in associations
                if a["tool_id"]["service_name"] == service_name
                and a["tool_id"]["tool_name"] == tool_name
            ),
            None,
        ):
            association_response = requests.patch(
                urljoin(ctx.obj.server_address, f"/agents/{agent_id}/guidelines/{guideline_id}"),
                json={
                    "tool_associations": {
                        "remove": [
                            {
                                "service_name": service_name,
                                "tool_name": tool_name,
                            }
                        ]
                    }
                },
            )

            association_response.raise_for_status()

            return association["id"]

        raise ValueError(
            f"An association between {guideline_id} and the tool {tool_name} from {service_name} was not found"
        )

    @staticmethod
    def list_variables(
        ctx: click.Context,
        agent_id: str,
    ) -> list[ContextVariableDTO]:
        response = requests.get(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/context-variables/")
        )

        response.raise_for_status()

        return cast(list[ContextVariableDTO], response.json()["context_variables"])

    @staticmethod
    def read_variable_by_name(
        ctx: click.Context,
        agent_id: str,
        name: str,
    ) -> ContextVariableDTO:
        variables = Actions.list_variables(ctx, agent_id)

        if variable := next((v for v in variables if v["name"] == name), None):
            return cast(ContextVariableDTO, variable)

        raise ValueError("A variable called '{name}' was not found under agent '{agent_id}'")

    @staticmethod
    def create_variable(
        ctx: click.Context,
        agent_id: str,
        name: str,
        description: str,
    ) -> ContextVariableDTO:
        response = requests.post(
            urljoin(ctx.obj.server_address, f"/agents/{agent_id}/context-variables"),
            json={
                "name": name,
                "description": description,
            },
        )

        response.raise_for_status()

        return cast(ContextVariableDTO, response.json()["context_variable"])

    @staticmethod
    def remove_variable(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
    ) -> None:
        response = requests.delete(
            urljoin(
                ctx.obj.server_address,
                f"/agents/{agent_id}/context-variables/{variable_id}",
            )
        )

        response.raise_for_status()

    @staticmethod
    def set_variable_value(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
        key: str,
        value: str,
    ) -> ContextVariableValueDTO:
        response = requests.put(
            urljoin(
                ctx.obj.server_address,
                f"/agents/{agent_id}/context-variables/{variable_id}/{key}",
            ),
            json={
                "data": value,
            },
        )

        response.raise_for_status()

        return cast(ContextVariableValueDTO, response.json()["context_variable_value"])

    @staticmethod
    def read_variable(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
        include_values: bool,
    ) -> ReadContextVariableResponse:
        response = requests.get(
            urljoin(
                ctx.obj.server_address,
                f"/agents/{agent_id}/context-variables/{variable_id}",
            ),
            params={"include_values": include_values},
        )

        response.raise_for_status()

        return cast(ReadContextVariableResponse, response.json())

    @staticmethod
    def read_variable_value(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
        key: str,
    ) -> ContextVariableValueDTO:
        response = requests.get(
            urljoin(
                ctx.obj.server_address,
                f"/agents/{agent_id}/context-variables/{variable_id}/{key}",
            )
        )

        response.raise_for_status()

        return cast(ContextVariableValueDTO, response.json())

    @staticmethod
    def add_service(
        ctx: click.Context,
        name: str,
        kind: str,
        url: str,
        source: str,
    ) -> ServiceDTO:
        if kind == "sdk":
            payload = {
                "kind": "sdk",
                "url": url,
            }
        elif kind == "openapi":
            payload = {
                "kind": "openapi",
                "url": url,
                "source": source,
            }
        else:
            raise ValueError(f"Unsupported kind: {kind}")

        response = requests.put(
            urljoin(ctx.obj.server_address, f"/services/{name}"),
            json=payload,
        )

        response.raise_for_status()

        return cast(ServiceDTO, response.json())

    @staticmethod
    def remove_service(
        ctx: click.Context,
        name: str,
    ) -> None:
        response = requests.delete(urljoin(ctx.obj.server_address, f"/services/{name}"))

        response.raise_for_status()

    @staticmethod
    def list_services(ctx: click.Context) -> list[ServiceDTO]:
        response = requests.get(urljoin(ctx.obj.server_address, "services"))

        response.raise_for_status()

        return cast(list[ServiceDTO], response.json()["services"])

    @staticmethod
    def read_service(ctx: click.Context, service_name: str) -> ServiceDTO:
        response = requests.get(urljoin(ctx.obj.server_address, f"/services/{service_name}"))

        response.raise_for_status()

        return cast(ServiceDTO, response.json())


class Interface:
    @staticmethod
    def _write_success(message: str) -> None:
        rich.print(Text(message, style="bold green"))

    @staticmethod
    def _write_error(message: str) -> None:
        rich.print(Text(message, style="bold red"), file=sys.stderr)

    @staticmethod
    def _print_table(
        data: Iterable[Any],
        **kwargs: Any,
    ) -> None:
        rich.print(
            tabulate(
                data,
                headers="keys",
                tablefmt="rounded_grid",
                **kwargs,
            )
        )

    @staticmethod
    def _render_agents(agents: list[AgentDTO]) -> None:
        agent_items = [
            {
                "ID": a["id"],
                "Name": a["name"],
                "Creation Date": format_datetime(a["creation_utc"]),
                "Description": a["description"] or "",
                "Max Engine Iterations": a["max_engine_iterations"],
            }
            for a in agents
        ]

        Interface._print_table(agent_items, maxcolwidths=[None, None, None, 60, None])

    @staticmethod
    def create_agent(
        ctx: click.Context,
        name: str,
        description: Optional[str],
        max_engine_iterations: Optional[int],
    ) -> None:
        try:
            agent = Actions.create_agent(ctx, name, description, max_engine_iterations)
            Interface._write_success(f"Added agent (id={agent['id']})")
            Interface._render_agents([agent])
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_agent(ctx: click.Context, agent_id: str) -> None:
        try:
            agent = Actions.read_agent(ctx, agent_id)
            Interface._render_agents([agent])
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def list_agents(ctx: click.Context) -> None:
        agents = Actions.list_agents(ctx)

        if not agents:
            rich.print("No data available")
            return

        Interface._render_agents(agents)

    @staticmethod
    def get_default_agent(ctx: click.Context) -> str:
        agents = Actions.list_agents(ctx)
        assert agents
        return str(agents[0]["id"])

    @staticmethod
    def patch_agent(
        ctx: click.Context,
        agent_id: str,
        description: Optional[str],
        max_engine_iterations: Optional[int],
    ) -> None:
        try:
            Actions.patch_agent(ctx, agent_id, description, max_engine_iterations)
            Interface._write_success(f"Updated agent (id={agent_id})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_sessions(sessions: list[SessionDTO]) -> None:
        session_items = [
            {
                "ID": s["id"],
                "Title": s["title"] or "",
                "Creation Date": format_datetime(s["creation_utc"]),
                "End User ID": s["end_user_id"],
            }
            for s in sessions
        ]

        Interface._print_table(session_items, maxcolwidths=[None, 40, None, None])

    @staticmethod
    def _render_events(events: list[EventDTO]) -> None:
        event_items = [
            {
                "ID": e["id"],
                "Creation Date": format_datetime(e["creation_utc"]),
                "Correlation ID": e["correlation_id"],
                "Source": e["source"],
                "Offset": e["offset"],
                "Kind": e["kind"],
                "Data": e["data"],
            }
            for e in events
        ]

        Interface._print_table(event_items, maxcolwidths=[None, None, None, None, None, 40])

    @staticmethod
    def view_session(
        ctx: click.Context,
        session_id: str,
    ) -> None:
        events = Actions.list_events(ctx, session_id)

        if not events:
            rich.print("No data available")
            return

        Interface._render_events(events=events)

    @staticmethod
    def list_sessions(
        ctx: click.Context,
        agent_id: Optional[str],
        end_user_id: Optional[str],
    ) -> None:
        sessions = Actions.list_sessions(ctx, agent_id, end_user_id)

        if not sessions:
            rich.print("No data available")
            return

        Interface._render_sessions(sessions)

    @staticmethod
    def create_session(
        ctx: click.Context,
        agent_id: str,
        end_user_id: str,
        title: Optional[str] = None,
    ) -> None:
        session = Actions.create_session(ctx, agent_id, end_user_id, title)
        Interface._write_success(f"Added session (id={session['id']})")
        Interface._render_sessions([session])

    @staticmethod
    def inspect_interaction(
        ctx: click.Context,
        session_id: str,
        correlation_id: str,
    ) -> None:
        interaction = Actions.inspect_interaction(ctx, session_id, correlation_id)

        rich.print(f"Session ID: '{session_id}'")
        rich.print(f"Correlation ID: '{correlation_id}'\n")

        for i, iteration in enumerate(interaction["preparation_iterations"]):
            rich.print(Text(f"Iteration #{i}:", style="bold yellow"))

            rich.print(Text(f"{INDENT}Guideline Propositions:", style="bold"))

            if iteration["guideline_propositions"]:
                for proposition in iteration["guideline_propositions"]:
                    rich.print(f"{INDENT*2}Predicate: {proposition['predicate']}")
                    rich.print(f"{INDENT*2}Action: {proposition['action']}")
                    rich.print(f"{INDENT*2}Relevance Score: {proposition['score']}/10")
                    rich.print(f"{INDENT*2}Rationale: {proposition['rationale']}\n")
            else:
                rich.print(f"{INDENT*2}(none)\n")

            rich.print(Text(f"{INDENT}Tool Calls:", style="bold"))

            if iteration["tool_calls"]:
                for tool_call in iteration["tool_calls"]:
                    rich.print(f"{INDENT*2}Tool Name: {tool_call['tool_name']}")
                    rich.print(f"{INDENT*2}Arguments: {tool_call['arguments']}")
                    rich.print(f"{INDENT*2}Result: {tool_call['result']}\n")
            else:
                rich.print(f"{INDENT*2}(none)\n")

            rich.print(Text(f"{INDENT}Context Variables:", style="bold"))

            if iteration["context_variables"]:
                for variable in iteration["context_variables"]:
                    rich.print(f"{INDENT*2}Name: {variable['name']}")
                    rich.print(f"{INDENT*2}Key: {variable['key']}")
                    rich.print(f"{INDENT*2}Value: {variable['value']}\n")
            else:
                rich.print(f"{INDENT*2}(none)\n")

            rich.print(Text(f"{INDENT}Glossary Terms:", style="bold"))

            if iteration["terms"]:
                for term in iteration["terms"]:
                    rich.print(f"{INDENT*2}Name: {term['name']}")
                    rich.print(f"{INDENT*2}Description: {term['description']}\n")
            else:
                rich.print(f"{INDENT*2}(none)\n")

    @staticmethod
    def create_event(
        ctx: click.Context,
        session_id: str,
        message: str,
    ) -> None:
        event = Actions.create_event(ctx, session_id, message)
        Interface._write_success(f"Added event (id={event['id']})")

    @staticmethod
    def chat(
        ctx: click.Context,
        session_id: str,
    ) -> None:
        def print_message(message_event: dict[str, Any]) -> None:
            role = {"end_user": "User", "ai_agent": "Agent"}[message_event["source"]]
            prefix = Text(
                f"{role}:".ljust(6),
                style="bold " + {"User": "blue", "Agent": "green"}[role],
            )

            message = wrap(
                message_event["data"]["message"],
                subsequent_indent=" " * (1 + len(prefix)),
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

                new_event = response.json()["event"]

                last_known_offset = new_event["offset"]

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
    def remove_term(
        ctx: click.Context,
        agent_id: str,
        name: str,
    ) -> None:
        Actions.remove_term(ctx, agent_id, name)
        Interface._write_success(f"Removed term '{name}'")

    @staticmethod
    def list_terms(
        ctx: click.Context,
        agent_id: str,
    ) -> None:
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
        tool_associations: list[GuidelineToolAssociationDTO],
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

        if tool_associations:
            rich.print("\nTool(s) Enabled:")
            Interface._render_guideline_tool_associations(tool_associations)

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
            guideline_with_connections_and_associations = Actions.create_guideline(
                ctx,
                agent_id,
                predicate,
                action,
                check,
                index,
            )

            guideline = guideline_with_connections_and_associations["guideline"]
            Interface._write_success(f"Added guideline (id={guideline['id']})")

            Interface._render_guideline_entailments(
                guideline_with_connections_and_associations["guideline"],
                guideline_with_connections_and_associations["connections"],
                guideline_with_connections_and_associations["tool_associations"],
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
                Text(
                    "\nTo force-add despite these errors, re-run with --no-check",
                    style="bold",
                )
            )
            set_exit_status(1)
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_guideline(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
    ) -> None:
        try:
            Actions.remove_guideline(ctx, agent_id, guideline_id)
            Interface._write_success(f"Removed guideline (id={guideline_id})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_guideline(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
    ) -> None:
        try:
            guideline_with_connections_and_associations = Actions.read_guideline(
                ctx, agent_id, guideline_id
            )

            Interface._render_guidelines([guideline_with_connections_and_associations["guideline"]])

            Interface._render_guideline_entailments(
                guideline_with_connections_and_associations["guideline"],
                guideline_with_connections_and_associations["connections"],
                guideline_with_connections_and_associations["tool_associations"],
                include_indirect=True,
            )
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def list_guidelines(
        ctx: click.Context,
        agent_id: str,
    ) -> None:
        try:
            guidelines = Actions.list_guidelines(ctx, agent_id)

            if not guidelines:
                rich.print("No data available")
                return

            Interface._render_guidelines(guidelines)

        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def create_entailment(
        ctx: click.Context,
        agent_id: str,
        source_guideline_id: str,
        target_guideline_id: str,
        kind: str,
    ) -> None:
        try:
            connection = Actions.add_entailment(
                ctx,
                agent_id,
                source_guideline_id,
                target_guideline_id,
                kind,
            )
            Interface._write_success(f"Added connection (id={connection["connections"][0]['id']})")
            Interface._print_table([connection])
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

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
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_guideline_tool_associations(
        associations: list[GuidelineToolAssociationDTO],
    ) -> None:
        association_items = [
            {
                "Association ID": a["id"],
                "Guideline ID": a["guideline_id"],
                "Service Name": a["tool_id"]["service_name"],
                "Tool Name": a["tool_id"]["tool_name"],
            }
            for a in associations
        ]

        Interface._print_table(association_items)

    @staticmethod
    def add_association(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        try:
            association = Actions.add_association(
                ctx, agent_id, guideline_id, service_name, tool_name
            )

            Interface._write_success(
                f"Enabled tool '{tool_name}' from service '{service_name}' for guideline '{guideline_id}'"
            )
            Interface._render_guideline_tool_associations(association["tool_associations"])

        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_association(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        try:
            association_id = Actions.remove_association(
                ctx, agent_id, guideline_id, service_name, tool_name
            )

            Interface._write_success(f"Removed tool association (id={association_id})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

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
    def _render_variable(variable: ContextVariableDTO) -> None:
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
    def list_variables(
        ctx: click.Context,
        agent_id: str,
    ) -> None:
        variables = Actions.list_variables(ctx, agent_id)

        if not variables:
            rich.print("No variables found")
            return

        variable_items = [
            {
                "ID": variable["id"],
                "Name": variable["name"],
                "Description": variable["description"] or "",
                "Service Name": variable["tool_id"]["service_name"] if variable["tool_id"] else "",
                "Tool Name": variable["tool_id"]["tool_name"] if variable["tool_id"] else "",
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
        Interface._render_variable(variable)

    @staticmethod
    def remove_variable(ctx: click.Context, agent_id: str, name: str) -> None:
        try:
            variable = Actions.read_variable_by_name(ctx, agent_id, name)
            Actions.remove_variable(ctx, agent_id, variable["id"])
            Interface._write_success(f"Removed variable '{name}'")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_variable_key_value_pairs(pairs: dict[str, ContextVariableValueDTO]) -> None:
        values_items = [
            {
                "ID": value["id"],
                "Key": key,
                "Value": value["data"],
                "Last Modified": format_datetime(value["last_modified"]),
            }
            for key, value in pairs.items()
        ]

        Interface._print_table(values_items, maxcolwidths=[None, None, 80, 30])

    @staticmethod
    def set_variable_value(
        ctx: click.Context,
        agent_id: str,
        variable_name: str,
        key: str,
        value: str,
    ) -> None:
        try:
            variable = Actions.read_variable_by_name(ctx, agent_id, variable_name)

            cv_value = Actions.set_variable_value(
                ctx=ctx,
                agent_id=agent_id,
                variable_id=variable["id"],
                key=key,
                value=value,
            )

            Interface._write_success(f"Added value (id={cv_value['id']})")
            Interface._render_variable_key_value_pairs({key: cv_value})
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_variable(
        ctx: click.Context,
        agent_id: str,
        name: str,
    ) -> None:
        try:
            variable = Actions.read_variable_by_name(ctx, agent_id, name)

            read_variable_response = Actions.read_variable(
                ctx,
                agent_id,
                variable["id"],
                include_values=True,
            )

            Interface._render_variable(read_variable_response["context_variable"])

            if not read_variable_response["key_value_pairs"]:
                rich.print("No values are available")
                return

            Interface._render_variable_key_value_pairs(
                read_variable_response["key_value_pairs"],
            )
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_variable_value(
        ctx: click.Context,
        agent_id: str,
        variable_name: str,
        key: str,
    ) -> None:
        try:
            variable = Actions.read_variable_by_name(ctx, agent_id, variable_name)
            value = Actions.read_variable_value(ctx, agent_id, variable["id"], key)
            Interface._render_variable_key_value_pairs({key: value})
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_service(
        ctx: click.Context,
        name: str,
        kind: str,
        url: str,
        source: str,
    ) -> None:
        try:
            result = Actions.add_service(ctx, name, kind, url, source)
            Interface._write_success(f"Added service '{name}'")
            Interface._print_table([result])
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_service(
        ctx: click.Context,
        name: str,
    ) -> None:
        try:
            Actions.remove_service(ctx, name)
            Interface._write_success(f"Removed service '{name}'")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def list_services(ctx: click.Context) -> None:
        services = Actions.list_services(ctx)

        if not services:
            rich.print("No services available")
            return

        service_items = [
            {
                "Name": service["name"],
                "Type": service["kind"],
                "Source": service["url"],
            }
            for service in services
        ]

        Interface._print_table(service_items)

    @staticmethod
    def view_service(
        ctx: click.Context,
        service_name: str,
    ) -> None:
        try:
            service = Actions.read_service(ctx, service_name)
            rich.print(Text("Name:", style="bold"), service["name"])
            rich.print(Text("Kind:", style="bold"), service["kind"])
            rich.print(Text("Source:", style="bold"), service["url"])

            if service["tools"]:
                rich.print(Text("Tools:", style="bold"))
                for tool in service["tools"]:
                    rich.print(Text("    Name:", style="bold"), tool["name"])
                    if tool["description"]:
                        rich.print(
                            Text("    Description:\n       ", style="bold"), tool["description"]
                        )

                    if tool["parameters"]:
                        rich.print(Text("    Parameters:", style="bold"))
                        for param_name, param_desc in tool["parameters"].items():
                            rich.print(Text(f"      - {param_name}:", style="bold"), end=" ")
                            rich.print(param_desc)

                        rich.print("\n")
            else:
                rich.print("\nNo tools available for this service.")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)


async def async_main() -> None:
    click_completion.init()

    @dataclass(frozen=True)
    class Config:
        server_address: str

    @click.group
    @click.option(
        "-s",
        "--server",
        type=str,
        help="Server address",
        metavar="ADDRESS[:PORT]",
        required=True,
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

    @agent.command("add", help="Add a new agent")
    @click.argument("name")
    @click.option("-d", "--description", type=str, help="Agent description", required=False)
    @click.option("--max-engine-iterations", type=int, help="Max engine iterations", required=False)
    @click.pass_context
    def agent_add(
        ctx: click.Context,
        name: str,
        description: Optional[str],
        max_engine_iterations: Optional[int],
    ) -> None:
        Interface.create_agent(
            ctx=ctx,
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
        )

    @agent.command("view", help="View agent information")
    @click.argument("agent_id")
    @click.pass_context
    def agent_view(ctx: click.Context, agent_id: str) -> None:
        Interface.view_agent(ctx, agent_id)

    @agent.command("list", help="List agents")
    @click.pass_context
    def agent_list(ctx: click.Context) -> None:
        Interface.list_agents(ctx)

    @agent.command("update", help="Update an agent's details")
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("-d", "--description", type=str, help="Agent description", required=False)
    @click.option("--max-engine-iterations", type=int, help="Max engine iterations", required=False)
    @click.pass_context
    def agent_update(
        ctx: click.Context,
        agent_id: str,
        description: Optional[str],
        max_engine_iterations: Optional[int],
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.patch_agent(ctx, agent_id, description, max_engine_iterations)

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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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

    @session.command("list", help="List all sessions")
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Filter by agent ID",
        metavar="ID",
        required=False,
    )
    @click.option(
        "-u",
        "--end-user-id",
        type=str,
        help="Filter by End User ID",
        metavar="ID",
        required=False,
    )
    @click.pass_context
    def session_list(
        ctx: click.Context, agent_id: Optional[str], end_user_id: Optional[str]
    ) -> None:
        Interface.list_sessions(ctx, agent_id, end_user_id)

    @session.command("view", help="View session content")
    @click.argument("session_id")
    @click.pass_context
    def session_view(ctx: click.Context, session_id: str) -> None:
        Interface.view_session(ctx, session_id)

    @session.command("inspect", help="Inspect an interaction from a session")
    @click.argument("session_id")
    @click.argument("correlation_id")
    @click.pass_context
    def session_inspect(ctx: click.Context, session_id: str, correlation_id: str) -> None:
        Interface.inspect_interaction(ctx, session_id, correlation_id)

    @session.command("post", help="Post user message to session")
    @click.argument("session_id")
    @click.argument("message")
    @click.pass_context
    def session_post(ctx: click.Context, session_id: str, message: str) -> None:
        Interface.create_event(ctx, session_id, message)

    @session.command("chat", help="Enter chat mode within the session")
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.argument("name", type=str)
    @click.argument("description", type=str)
    @click.option(
        "-s",
        "--synonyms",
        type=str,
        help="Comma-separated list of synonyms",
        required=False,
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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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

    @guideline.command("enable-tool", help="Enable a tool for a guideline")
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.argument("guideline_id", type=str)
    @click.argument("service_name", type=str)
    @click.argument("tool_name", type=str)
    @click.pass_context
    def guideline_enable_tool(
        ctx: click.Context,
        agent_id: Optional[str],
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.add_association(
            ctx=ctx,
            agent_id=agent_id,
            guideline_id=guideline_id,
            service_name=service_name,
            tool_name=tool_name,
        )

    @guideline.command("disable-tool", help="Disable a tool for a guideline")
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.argument("guideline_id", type=str)
    @click.argument("service_name", type=str)
    @click.argument("tool_name", type=str)
    @click.pass_context
    def guideline_disable_tool(
        ctx: click.Context,
        agent_id: Optional[str],
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.remove_association(
            ctx=ctx,
            agent_id=agent_id,
            guideline_id=guideline_id,
            service_name=service_name,
            tool_name=tool_name,
        )

    @cli.group(help="Manage an agent's context variables")
    def variable() -> None:
        pass

    @variable.command("list", help="List all variables for an agent")
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
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

    @variable.command("set", help="Set a variable's value")
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.argument("name", type=str)
    @click.argument("key", type=str)
    @click.argument("value", type=str)
    @click.pass_context
    def variable_set(
        ctx: click.Context,
        agent_id: Optional[str],
        name: str,
        key: str,
        value: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.set_variable_value(
            ctx=ctx,
            agent_id=agent_id,
            variable_name=name,
            key=key,
            value=value,
        )

    @variable.command("get", help="Get the value(s) of a variable")
    @click.option(
        "-a",
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.argument("name", type=str)
    @click.argument("key", type=str, required=False)
    @click.pass_context
    def variable_get(
        ctx: click.Context,
        agent_id: Optional[str],
        name: str,
        key: Optional[str],
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        if key:
            Interface.view_variable_value(
                ctx=ctx,
                agent_id=agent_id,
                variable_name=name,
                key=key,
            )
        else:
            Interface.view_variable(
                ctx=ctx,
                agent_id=agent_id,
                name=name,
            )

    @cli.group(help="Manage services")
    def service() -> None:
        pass

    @service.command("add", help="Add a new service")
    @click.option(
        "-k",
        "--kind",
        type=click.Choice(["sdk", "openapi"], case_sensitive=False),
        required=True,
        help="Service kind",
    )
    @click.option(
        "-u",
        "--url",
        metavar="URL",
        required=True,
        help="Service root URL",
    )
    @click.option(
        "-s",
        "--source",
        required=False,
        metavar="SOURCE",
        help="For an OpenAPI service, this is the local path or URL to its openapi.json",
    )
    @click.argument("name", type=str)
    @click.pass_context
    def service_add(
        ctx: click.Context,
        name: str,
        kind: str,
        url: str,
        source: str,
    ) -> None:
        Interface.add_service(ctx, name, kind, url, source)

    @service.command("remove", help="Remove a service")
    @click.argument("name", type=str)
    @click.pass_context
    def service_remove(ctx: click.Context, name: str) -> None:
        Interface.remove_service(ctx, name)

    @service.command("list", help="List all services")
    @click.pass_context
    def service_list(ctx: click.Context) -> None:
        Interface.list_services(ctx)

    @service.command("view", help="View a specific service and its tools")
    @click.argument("name", type=str)
    @click.pass_context
    def service_view(ctx: click.Context, name: str) -> None:
        Interface.view_service(ctx, name)

    cli()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
    sys.exit(get_exit_status())
