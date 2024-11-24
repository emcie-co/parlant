# mypy: disable-error-code=import-untyped

import asyncio
import click
import click.shell_completion
import click_completion  # type: ignore
from dataclasses import dataclass
from datetime import datetime
import requests
import rich
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TaskProgressColumn
from rich import box
from rich.table import Table
from rich.text import Text
import sys
import time
from typing import Any, Optional, cast

from parlant.client import ParlantClient
from parlant.client.types import (
    Agent,
    ContextVariable,
    ContextVariableReadResult,
    ContextVariableValue,
    Event,
    EventInspectionResult,
    FreshnessRules,
    Guideline,
    GuidelineConnection,
    GuidelineConnectionAddition,
    GuidelineConnectionUpdateParams,
    GuidelineContent,
    GuidelinePayload,
    GuidelineToolAssociation,
    GuidelineToolAssociationUpdateParams,
    GuidelineWithConnectionsAndToolAssociations,
    OpenApiServiceParams,
    Payload,
    SdkServiceParams,
    Service,
    Session,
    Term,
    ToolId,
    Customer,
    CustomerExtraUpdateParams,
    TagsUpdateParams,
    Tag,
    ConsumptionOffsetsUpdateParams,
)

INDENT = "  "


class CoherenceCheckFailure(Exception):
    def __init__(self, contradictions: list[dict[str, Any]]) -> None:
        self.contradictions = contradictions


def format_datetime(datetime_str: str) -> str:
    return datetime.fromisoformat(datetime_str).strftime("%Y-%m-%d %I:%M:%S %p %Z")


def reformat_datetime(datetime: datetime) -> str:
    return datetime.strftime("%Y-%m-%d %I:%M:%S %p %Z")


_EXIT_STATUS = 0


def get_exit_status() -> int:
    return _EXIT_STATUS


def set_exit_status(status: int) -> None:
    global _EXIT_STATUS
    _EXIT_STATUS = status  # type: ignore


class Actions:
    @staticmethod
    def create_agent(
        ctx: click.Context,
        name: str,
        description: Optional[str],
        max_engine_iterations: Optional[int],
    ) -> Agent:
        client = cast(ParlantClient, ctx.obj.client)

        return client.agents.create(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
        )

    @staticmethod
    def delete_agent(
        ctx: click.Context,
        agent_id: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.agents.delete(agent_id=agent_id)

    @staticmethod
    def view_agent(
        ctx: click.Context,
        agent_id: str,
    ) -> Agent:
        client = cast(ParlantClient, ctx.obj.client)

        return client.agents.retrieve(agent_id)

    @staticmethod
    def list_agents(ctx: click.Context) -> list[Agent]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.agents.list()

    @staticmethod
    def update_agent(
        ctx: click.Context,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_engine_iterations: Optional[int] = None,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)

        client.agents.update(
            agent_id,
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
        )

    @staticmethod
    def create_session(
        ctx: click.Context,
        agent_id: str,
        customer_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Session:
        client = cast(ParlantClient, ctx.obj.client)

        return client.sessions.create(
            agent_id=agent_id,
            customer_id=customer_id,
            allow_greeting=False,
            title=title,
        )

    @staticmethod
    def delete_session(ctx: click.Context, session_id: str) -> None:
        client = cast(ParlantClient, ctx.obj.client)

        client.sessions.delete(session_id)

    @staticmethod
    def update_session(
        ctx: click.Context,
        session_id: str,
        consumption_offsets: Optional[int] = None,
        title: Optional[str] = None,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)

        if consumption_offsets:
            client.sessions.update(
                session_id=session_id,
                consumption_offsets=ConsumptionOffsetsUpdateParams(client=consumption_offsets),
                title=title,
            )
        else:
            client.sessions.update(
                session_id=session_id,
                title=title,
            )

    @staticmethod
    def list_sessions(
        ctx: click.Context,
        agent_id: Optional[str],
        customer_id: Optional[str],
    ) -> list[Session]:
        client = cast(ParlantClient, ctx.obj.client)

        return client.sessions.list(
            agent_id=agent_id,
            customer_id=customer_id,
        )

    @staticmethod
    def inspect_event(
        ctx: click.Context,
        session_id: str,
        event_id: str,
    ) -> EventInspectionResult:
        client = cast(ParlantClient, ctx.obj.client)

        return client.sessions.inspect_event(
            session_id=session_id,
            event_id=event_id,
        )

    @staticmethod
    def list_events(
        ctx: click.Context,
        session_id: str,
    ) -> list[Event]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.sessions.list_events(session_id=session_id)

    @staticmethod
    def create_event(
        ctx: click.Context,
        session_id: str,
        message: str,
    ) -> Event:
        client = cast(ParlantClient, ctx.obj.client)

        return client.sessions.create_event(
            session_id,
            kind="message",
            source="customer",
            data=message,
        )

    @staticmethod
    def create_term(
        ctx: click.Context,
        agent_id: str,
        name: str,
        description: str,
        synonyms: list[str],
    ) -> Term:
        client = cast(ParlantClient, ctx.obj.client)

        return client.glossary.create_term(
            agent_id,
            name=name,
            description=description,
            synonyms=synonyms,
        )

    @staticmethod
    def update_term(
        ctx: click.Context,
        agent_id: str,
        term_id: str,
        name: Optional[str],
        description: Optional[str],
        synonyms: list[str],
    ) -> Term:
        client = cast(ParlantClient, ctx.obj.client)

        return client.glossary.update_term(
            agent_id,
            term_id,
            name=name,
            description=description,
            synonyms=synonyms,
        )

    @staticmethod
    def delete_term(
        ctx: click.Context,
        agent_id: str,
        term_id: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.glossary.delete_term(agent_id, term_id)

    @staticmethod
    def list_terms(
        ctx: click.Context,
        agent_id: str,
    ) -> list[Term]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.glossary.list_terms(agent_id)

    @staticmethod
    def create_guideline(
        ctx: click.Context,
        agent_id: str,
        condition: str,
        action: str,
        check: bool,
        index: bool,
        updated_id: Optional[str] = None,
    ) -> GuidelineWithConnectionsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)

        evaluation = client.evaluations.create(
            agent_id=agent_id,
            payloads=[
                Payload(
                    kind="guideline",
                    guideline=GuidelinePayload(
                        content=GuidelineContent(
                            condition=condition,
                            action=action,
                        ),
                        operation="add",
                        updated_id=updated_id,
                        coherence_check=check,
                        connection_proposition=index,
                    ),
                ),
            ],
        )

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(style="bold blue"),
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
        ) as progress:
            progress_task = progress.add_task("Evaluating guideline impact\n", total=100)
            while True:
                time.sleep(0.5)
                evaluation_result = client.evaluations.retrieve(evaluation.id)

                if evaluation_result.status in ["pending", "running"]:
                    progress.update(progress_task, completed=int(evaluation_result.progress))
                    continue

                if evaluation_result.status == "completed":
                    progress.update(progress_task, completed=100)

                    invoice = evaluation_result.invoices[0]
                    if invoice.approved:
                        assert invoice.data
                        assert invoice.data.guideline
                        assert invoice.payload.guideline

                        guideline_result = client.guidelines.create(
                            agent_id,
                            invoices=[invoice],
                        )
                        return guideline_result.items[0]

                    else:
                        assert invoice.data
                        assert invoice.data.guideline
                        contradictions = list(
                            map(lambda x: x.__dict__, invoice.data.guideline.coherence_checks)
                        )
                        raise CoherenceCheckFailure(contradictions=contradictions)

                elif evaluation_result.status == "failed":
                    raise ValueError(evaluation_result.error)

    @staticmethod
    def update_guideline(
        ctx: click.Context,
        agent_id: str,
        condition: str,
        action: str,
        check: bool,
        index: bool,
        updated_id: Optional[str] = None,
    ) -> GuidelineWithConnectionsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)

        evaluation = client.evaluations.create(
            agent_id=agent_id,
            payloads=[
                Payload(
                    kind="guideline",
                    guideline=GuidelinePayload(
                        content=GuidelineContent(
                            condition=condition,
                            action=action,
                        ),
                        operation="update",
                        updated_id=updated_id,
                        coherence_check=check,
                        connection_proposition=index,
                    ),
                ),
            ],
        )

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(style="bold blue"),
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
        ) as progress:
            progress_task = progress.add_task("Evaluating guideline impact\n", total=100)

            while True:
                time.sleep(0.5)
                evaluation_result = client.evaluations.retrieve(evaluation.id)

                if evaluation_result.status in ["pending", "running"]:
                    progress.update(progress_task, completed=int(evaluation_result.progress))
                    continue

                if evaluation_result.status == "completed":
                    progress.update(progress_task, completed=100)
                    invoice = evaluation_result.invoices[0]
                    if invoice.approved:
                        assert invoice.data
                        assert invoice.data.guideline
                        assert invoice.payload.guideline

                        guideline_result = client.guidelines.create(
                            agent_id,
                            invoices=[invoice],
                        )
                        return guideline_result.items[0]

                    else:
                        assert invoice.data
                        assert invoice.data.guideline
                        contradictions = list(
                            map(lambda x: x.__dict__, invoice.data.guideline.coherence_checks)
                        )
                        raise CoherenceCheckFailure(contradictions=contradictions)

                elif evaluation_result.status == "failed":
                    raise ValueError(evaluation_result.error)

    @staticmethod
    def delete_guideline(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.guidelines.delete(agent_id, guideline_id)

    @staticmethod
    def view_guideline(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
    ) -> GuidelineWithConnectionsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)
        return client.guidelines.retrieve(agent_id, guideline_id)

    @staticmethod
    def list_guidelines(
        ctx: click.Context,
        agent_id: str,
    ) -> list[Guideline]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.guidelines.list(agent_id)

    @staticmethod
    def create_entailment(
        ctx: click.Context,
        agent_id: str,
        source_guideline_id: str,
        target_guideline_id: str,
        kind: str,
    ) -> GuidelineWithConnectionsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)

        return client.guidelines.update(
            agent_id,
            source_guideline_id,
            connections=GuidelineConnectionUpdateParams(
                add=[
                    GuidelineConnectionAddition(
                        source=source_guideline_id,
                        target=target_guideline_id,
                        kind=kind,
                    ),
                ]
            ),
        )

    @staticmethod
    def remove_entailment(
        ctx: click.Context,
        agent_id: str,
        source_guideline_id: str,
        target_guideline_id: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        guideline_result = client.guidelines.retrieve(agent_id, source_guideline_id)
        connections = guideline_result.connections

        if connection := next(
            (c for c in connections if target_guideline_id in [c.source.id, c.target.id]),
            None,
        ):
            client.guidelines.update(
                agent_id,
                source_guideline_id,
                connections=GuidelineConnectionUpdateParams(remove=[target_guideline_id]),
            )

            return connection.id

        raise ValueError(
            f"An entailment between {source_guideline_id} and {target_guideline_id} was not found"
        )

    @staticmethod
    def add_guideline_tool_association(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> GuidelineWithConnectionsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)

        return client.guidelines.update(
            agent_id,
            guideline_id,
            tool_associations=GuidelineToolAssociationUpdateParams(
                add=[
                    ToolId(
                        service_name=service_name,
                        tool_name=tool_name,
                    ),
                ]
            ),
        )

    @staticmethod
    def remove_guideline_tool_association(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        guideline_result = client.guidelines.retrieve(agent_id, guideline_id)
        associations = guideline_result.tool_associations

        if association := next(
            (
                a
                for a in associations
                if a.tool_id.service_name == service_name and a.tool_id.tool_name == tool_name
            ),
            None,
        ):
            client.guidelines.update(
                agent_id,
                guideline_id,
                tool_associations=GuidelineToolAssociationUpdateParams(
                    remove=[
                        ToolId(
                            service_name=service_name,
                            tool_name=tool_name,
                        ),
                    ]
                ),
            )

            return association.id

        raise ValueError(
            f"An association between {guideline_id} and the tool {tool_name} from {service_name} was not found"
        )

    @staticmethod
    def list_variables(
        ctx: click.Context,
        agent_id: str,
    ) -> list[ContextVariable]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.context_variables.list(agent_id)

    @staticmethod
    def view_variable(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
    ) -> ContextVariable:
        client = cast(ParlantClient, ctx.obj.client)

        variables = client.context_variables.list(agent_id)

        if variable := next((v for v in variables if v.id == variable_id), None):
            return variable

        raise ValueError("A variable called '{name}' was not found under agent '{agent_id}'")

    @staticmethod
    def create_variable(
        ctx: click.Context,
        agent_id: str,
        name: str,
        description: str,
    ) -> ContextVariable:
        client = cast(ParlantClient, ctx.obj.client)

        return client.context_variables.create(
            agent_id,
            name=name,
            description=description,
        )

    @staticmethod
    def delete_variable(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.context_variables.delete(agent_id, variable_id)

    @staticmethod
    def set_variable_value(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
        key: str,
        value: str,
    ) -> ContextVariableValue:
        client = cast(ParlantClient, ctx.obj.client)

        return client.context_variables.set_value(
            agent_id,
            variable_id,
            key,
            data=value,
        )

    @staticmethod
    def read_variable(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
        include_values: bool,
    ) -> ContextVariableReadResult:
        client = cast(ParlantClient, ctx.obj.client)

        return client.context_variables.retrieve(
            agent_id,
            variable_id,
            include_values=include_values,
        )

    @staticmethod
    def read_variable_value(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
        key: str,
    ) -> ContextVariableValue:
        client = cast(ParlantClient, ctx.obj.client)

        return client.context_variables.get_value(
            agent_id,
            variable_id,
            key,
        )

    @staticmethod
    def create_service(
        ctx: click.Context,
        name: str,
        kind: str,
        url: str,
        source: str,
    ) -> Service:
        client = cast(ParlantClient, ctx.obj.client)

        if kind == "sdk":
            result = client.services.create_or_update(
                name=name,
                kind="sdk",
                sdk=SdkServiceParams(url=url),
            )

        elif kind == "openapi":
            result = client.services.create_or_update(
                name=name,
                kind="openapi",
                openapi=OpenApiServiceParams(url=url, source=source),
            )

        else:
            raise ValueError(f"Unsupported kind: {kind}")

        return Service(
            name=result.name,
            kind=result.kind,
            url=result.url,
        )

    @staticmethod
    def delete_service(
        ctx: click.Context,
        name: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.services.delete(name)

    @staticmethod
    def list_services(ctx: click.Context) -> list[Service]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.services.list()

    @staticmethod
    def view_service(
        ctx: click.Context,
        service_name: str,
    ) -> Service:
        client = cast(ParlantClient, ctx.obj.client)
        return client.services.retrieve(service_name)

    @staticmethod
    def list_customers(
        ctx: click.Context,
    ) -> list[Customer]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.customers.list()

    @staticmethod
    def create_customer(
        ctx: click.Context,
        name: str,
    ) -> Customer:
        client = cast(ParlantClient, ctx.obj.client)
        return client.customers.create(name=name, extra={})

    @staticmethod
    def delete_customer(
        ctx: click.Context,
        customer_id: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.customers.delete(customer_id)

    @staticmethod
    def view_customer(
        ctx: click.Context,
        customer_id: str,
    ) -> Customer:
        client = cast(ParlantClient, ctx.obj.client)
        result = client.customers.retrieve(customer_id=customer_id)
        return result

    @staticmethod
    def add_customer_extra(
        ctx: click.Context,
        customer_id: str,
        key: str,
        value: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.customers.update(
            customer_id=customer_id, extra=CustomerExtraUpdateParams(add={key: value})
        )

    @staticmethod
    def remove_customer_extra(
        ctx: click.Context,
        customer_id: str,
        key: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.customers.update(
            customer_id=customer_id, extra=CustomerExtraUpdateParams(remove=[key])
        )

    @staticmethod
    def add_customer_tag(ctx: click.Context, customer_id: str, tag_id: str) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.customers.update(customer_id=customer_id, tags=TagsUpdateParams(add=[tag_id]))

    @staticmethod
    def remove_customer_tag(ctx: click.Context, customer_id: str, tag_id: str) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.customers.update(customer_id=customer_id, tags=TagsUpdateParams(remove=[tag_id]))

    @staticmethod
    def list_tags(ctx: click.Context) -> list[Tag]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.tags.list()

    @staticmethod
    def create_tag(ctx: click.Context, name: str) -> Tag:
        client = cast(ParlantClient, ctx.obj.client)
        return client.tags.create(name=name)

    @staticmethod
    def view_tag(ctx: click.Context, tag_id: str) -> Tag:
        client = cast(ParlantClient, ctx.obj.client)
        return client.tags.retrieve(tag_id=tag_id)

    @staticmethod
    def update_tag(ctx: click.Context, tag_id: str, name: str) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.tags.update(tag_id=tag_id, name=name)

    @staticmethod
    def delete_tag(ctx: click.Context, tag_id: str) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.tags.delete(tag_id=tag_id)


def raise_for_status_with_detail(response: requests.Response) -> None:
    """Raises :class:`HTTPError`, if one occurred, with detail if exists

    Adapted from requests.Response.raise_for_status"""
    http_error_msg = ""

    if isinstance(response.reason, bytes):
        try:
            reason = response.reason.decode("utf-8")
        except UnicodeDecodeError:
            reason = response.reason.decode("iso-8859-1")
    else:
        reason = response.reason

    if 400 <= response.status_code < 500:
        http_error_msg = (
            f"{response.status_code} Client Error: {reason} for url: {response.url}"
        ) + (f": {response.json()["detail"]}" if "detail" in response.json() else "")
    elif 500 <= response.status_code < 600:
        http_error_msg = (
            f"{response.status_code} Server Error: {reason} for url: {response.url}"
            + (f": {response.json()["detail"]}" if "detail" in response.json() else "")
        )

    if http_error_msg:
        raise requests.HTTPError(http_error_msg, response=response)


class Interface:
    @staticmethod
    def _write_success(message: str) -> None:
        rich.print(Text(message, style="bold green"))

    @staticmethod
    def _write_error(message: str) -> None:
        rich.print(Text(message, style="bold red"), file=sys.stderr)

    @staticmethod
    def _print_table(data: list[dict[str, Any]]) -> None:
        table = Table(box=box.ROUNDED, border_style="bright_green")

        headers = list(data[0].keys())

        for header in headers:
            table.add_column(header, header_style="bright_green", overflow="fold")

        for row in data:
            table.add_row(*list(map(str, row.values())))

        rich.print(table)

    @staticmethod
    def _render_agents(agents: list[Agent]) -> None:
        agent_items: list[dict[str, Any]] = [
            {
                "ID": a.id,
                "Name": a.name,
                "Creation Date": reformat_datetime(a.creation_utc),
                "Description": a.description or "",
                "Max Engine Iterations": a.max_engine_iterations,
            }
            for a in agents
        ]

        Interface._print_table(agent_items)

    @staticmethod
    def create_agent(
        ctx: click.Context,
        name: str,
        description: Optional[str],
        max_engine_iterations: Optional[int],
    ) -> None:
        try:
            agent = Actions.create_agent(ctx, name, description, max_engine_iterations)

            Interface._write_success(f"Added agent (id={agent.id})")
            Interface._render_agents([agent])
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_agent(ctx: click.Context, agent_id: str) -> None:
        try:
            Actions.delete_agent(ctx, agent_id=agent_id)
            Interface._write_success(f"Removed agent (id={agent_id})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_agent(ctx: click.Context, agent_id: str) -> None:
        try:
            agent = Actions.view_agent(ctx, agent_id)

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
        return str(agents[0].id)

    @staticmethod
    def update_agent(
        ctx: click.Context,
        agent_id: str,
        name: Optional[str],
        description: Optional[str],
        max_engine_iterations: Optional[int],
    ) -> None:
        try:
            Actions.update_agent(ctx, agent_id, name, description, max_engine_iterations)
            Interface._write_success(f"Updated agent (id={agent_id})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_sessions(sessions: list[Session]) -> None:
        session_items = [
            {
                "ID": s.id,
                "Title": s.title or "",
                "Creation Date": reformat_datetime(s.creation_utc),
                "Costumer ID": s.customer_id,
            }
            for s in sessions
        ]

        Interface._print_table(session_items)

    @staticmethod
    def _render_events(events: list[Event]) -> None:
        event_items: list[dict[str, Any]] = [
            {
                "Event ID": e.id,
                "Creation Date": reformat_datetime(e.creation_utc),
                "Correlation ID": e.correlation_id,
                "Source": e.source,
                "Offset": e.offset,
                "Kind": e.kind,
                "Data": e.data,
            }
            for e in events
        ]

        Interface._print_table(event_items)

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
        customer_id: Optional[str],
    ) -> None:
        sessions = Actions.list_sessions(ctx, agent_id, customer_id)

        if not sessions:
            rich.print("No data available")
            return

        Interface._render_sessions(sessions)

    @staticmethod
    def create_session(
        ctx: click.Context,
        agent_id: str,
        customer_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        session = Actions.create_session(ctx, agent_id, customer_id, title)
        Interface._write_success(f"Added session (id={session.id})")
        Interface._render_sessions([session])

    @staticmethod
    def delete_session(ctx: click.Context, session_id: str) -> None:
        try:
            Actions.delete_session(ctx, session_id=session_id)
            Interface._write_success(f"Removed session (id={session_id})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def update_session(
        ctx: click.Context,
        session_id: str,
        title: Optional[str] = None,
        consumption_offsets: Optional[int] = None,
    ) -> None:
        Actions.update_session(ctx, session_id, consumption_offsets, title)
        Interface._write_success(f"Updated session (id={session_id})")

    @staticmethod
    def inspect_event(
        ctx: click.Context,
        session_id: str,
        event_id: str,
    ) -> None:
        inspection = Actions.inspect_event(ctx, session_id, event_id)

        rich.print(f"Session ID: '{session_id}'")
        rich.print(f"Event ID: '{event_id}'\n")

        Interface._render_events([inspection.event])

        rich.print("\n")

        if not inspection.trace:
            return

        for i, iteration in enumerate(inspection.trace.preparation_iterations):
            rich.print(Text(f"Iteration #{i}:", style="bold yellow"))

            rich.print(Text(f"{INDENT}Guideline Propositions:", style="bold"))

            if iteration.guideline_propositions:
                for proposition in iteration.guideline_propositions:
                    rich.print(f"{INDENT*2}Condition: {proposition.condition}")
                    rich.print(f"{INDENT*2}Action: {proposition.action}")
                    rich.print(f"{INDENT*2}Relevance Score: {proposition.score}/10")
                    rich.print(f"{INDENT*2}Rationale: {proposition.rationale}\n")
            else:
                rich.print(f"{INDENT*2}(none)\n")

            rich.print(Text(f"{INDENT}Tool Calls:", style="bold"))

            if iteration.tool_calls:
                for tool_call in iteration.tool_calls:
                    rich.print(f"{INDENT*2}Tool Id: {tool_call.tool_id}")
                    rich.print(f"{INDENT*2}Arguments: {tool_call.arguments}")
                    rich.print(f"{INDENT*2}Result: {tool_call.result}\n")
            else:
                rich.print(f"{INDENT*2}(none)\n")

            rich.print(Text(f"{INDENT}Context Variables:", style="bold"))

            if iteration.context_variables:
                for variable in iteration.context_variables:
                    rich.print(f"{INDENT*2}Name: {variable.name}")
                    rich.print(f"{INDENT*2}Key: {variable.key}")
                    rich.print(f"{INDENT*2}Value: {variable.value}\n")
            else:
                rich.print(f"{INDENT*2}(none)\n")

            rich.print(Text(f"{INDENT}Glossary Terms:", style="bold"))

            if iteration.terms:
                for term in iteration.terms:
                    rich.print(f"{INDENT*2}Name: {term.name}")
                    rich.print(f"{INDENT*2}Description: {term.description}\n")
            else:
                rich.print(f"{INDENT*2}(none)\n")

    @staticmethod
    def create_event(
        ctx: click.Context,
        session_id: str,
        message: str,
    ) -> None:
        event = Actions.create_event(ctx, session_id, message)

        Interface._write_success(f"Added event (id={event.id})")

    @staticmethod
    def create_term(
        ctx: click.Context,
        agent_id: str,
        name: str,
        description: str,
        synonyms: list[str],
    ) -> None:
        term = Actions.create_term(
            ctx,
            agent_id,
            name,
            description,
            synonyms,
        )

        Interface._write_success(f"Added term (id={term.id})")
        Interface._print_table([term.__dict__])

    @staticmethod
    def update_term(
        ctx: click.Context,
        agent_id: str,
        term_id: str,
        name: Optional[str],
        description: Optional[str],
        synonyms: list[str],
    ) -> None:
        if not name and not description and not synonyms:
            Interface._write_error(
                "Error: No updates provided. Please provide at least one of the following: name, description, or synonyms to update the term."
            )
            return

        term = Actions.update_term(
            ctx,
            agent_id,
            term_id,
            name,
            description,
            synonyms,
        )
        Interface._write_success(f"Updated term (id={term.id})")
        Interface._print_table([term.__dict__])

    @staticmethod
    def delete_term(
        ctx: click.Context,
        agent_id: str,
        term_id: str,
    ) -> None:
        Actions.delete_term(ctx, agent_id, term_id)

        Interface._write_success(f"Removed term '{term_id}'")

    @staticmethod
    def list_terms(
        ctx: click.Context,
        agent_id: str,
    ) -> None:
        terms = Actions.list_terms(ctx, agent_id)

        if not terms:
            rich.print("No data available")
            return

        Interface._print_table(list(map(lambda t: t.__dict__, terms)))

    @staticmethod
    def _render_guidelines(guidelines: list[Guideline]) -> None:
        guideline_items: list[dict[str, Any]] = [
            {
                "ID": guideline.id,
                "Condition": guideline.condition,
                "Action": guideline.action,
            }
            for guideline in guidelines
        ]

        Interface._print_table(guideline_items)

    @staticmethod
    def _render_guideline_entailments(
        guideline: Guideline,
        connections: list[GuidelineConnection],
        tool_associations: list[GuidelineToolAssociation],
        include_indirect: bool,
    ) -> None:
        def to_direct_entailment_item(conn: GuidelineConnection) -> dict[str, str]:
            peer = conn.target if conn.source.id == guideline.id else conn.source

            return {
                "Connection ID": conn.id,
                "Entailment": "Strict" if conn.kind == "entails" else "Suggestive",
                "Role": "Source" if conn.source.id == guideline.id else "Target",
                "Peer Role": "Target" if conn.source.id == guideline.id else "Source",
                "Peer ID": peer.id,
                "Peer Condition": peer.condition,
                "Peer Action": peer.action,
            }

        def to_indirect_entailment_item(conn: GuidelineConnection) -> dict[str, str]:
            return {
                "Connection ID": conn.id,
                "Entailment": "Strict" if conn.kind == "entails" else "Suggestive",
                "Source ID": conn.source.id,
                "Source Condition": conn.source.condition,
                "Source Action": conn.source.action,
                "Target ID": conn.target.id,
                "Target Condition": conn.target.condition,
                "Target Action": conn.target.action,
            }

        if connections:
            direct = [c for c in connections if not c.indirect]
            indirect = [c for c in connections if c.indirect]

            if direct:
                rich.print("\nDirect Entailments:")
                Interface._print_table(list(map(lambda c: to_direct_entailment_item(c), direct)))

            if indirect and include_indirect:
                rich.print("\nIndirect Entailments:")
                Interface._print_table(
                    list(map(lambda c: to_indirect_entailment_item(c), indirect))
                )

        if tool_associations:
            rich.print("\nTool(s) Enabled:")
            Interface._render_guideline_tool_associations(tool_associations)

    @staticmethod
    def create_guideline(
        ctx: click.Context,
        agent_id: str,
        condition: str,
        action: str,
        check: bool,
        index: bool,
    ) -> None:
        try:
            guideline_with_connections_and_associations = Actions.create_guideline(
                ctx,
                agent_id,
                condition,
                action,
                check,
                index,
            )

            guideline = guideline_with_connections_and_associations.guideline
            Interface._write_success(f"Added guideline (id={guideline.id})")
            Interface._render_guideline_entailments(
                guideline_with_connections_and_associations.guideline,
                guideline_with_connections_and_associations.connections,
                guideline_with_connections_and_associations.tool_associations,
                include_indirect=False,
            )

        except CoherenceCheckFailure as e:
            contradictions = e.contradictions
            Interface._write_error("Failed to add guideline")
            rich.print("Detected potential incoherence with other guidelines:")
            Interface._print_table(contradictions)
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
    def update_guideline(
        ctx: click.Context,
        agent_id: str,
        condition: str,
        action: str,
        guideline_id: str,
        check: bool,
        index: bool,
    ) -> None:
        try:
            guideline_with_connections = Actions.update_guideline(
                ctx,
                agent_id=agent_id,
                condition=condition,
                action=action,
                check=check,
                index=index,
                updated_id=guideline_id,
            )

            guideline = guideline_with_connections.guideline
            Interface._write_success(f"Updated guideline (id={guideline.id})")
            Interface._render_guideline_entailments(
                guideline_with_connections.guideline,
                guideline_with_connections.connections,
                guideline_with_connections.tool_associations,
                include_indirect=False,
            )

        except CoherenceCheckFailure as e:
            contradictions = e.contradictions
            Interface._write_error("Failed to update guideline")
            rich.print("Detected potential incoherence with other guidelines:")
            Interface._print_table(contradictions)
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
    def delete_guideline(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
    ) -> None:
        try:
            Actions.delete_guideline(ctx, agent_id, guideline_id)

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
            guideline_with_connections_and_associations = Actions.view_guideline(
                ctx, agent_id, guideline_id
            )

            Interface._render_guidelines([guideline_with_connections_and_associations.guideline])
            Interface._render_guideline_entailments(
                guideline_with_connections_and_associations.guideline,
                guideline_with_connections_and_associations.connections,
                guideline_with_connections_and_associations.tool_associations,
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
            connection = Actions.create_entailment(
                ctx,
                agent_id,
                source_guideline_id,
                target_guideline_id,
                kind,
            )

            Interface._write_success(f"Added connection (id={connection.connections[0].id})")
            Interface._print_table([connection.dict()])
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
        associations: list[GuidelineToolAssociation],
    ) -> None:
        association_items = [
            {
                "Association ID": a.id,
                "Guideline ID": a.guideline_id,
                "Service Name": a.tool_id.service_name,
                "Tool Name": a.tool_id.tool_name,
            }
            for a in associations
        ]

        Interface._print_table(association_items)

    @staticmethod
    def add_guideline_tool_association(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        try:
            association = Actions.add_guideline_tool_association(
                ctx, agent_id, guideline_id, service_name, tool_name
            )

            Interface._write_success(
                f"Enabled tool '{tool_name}' from service '{service_name}' for guideline '{guideline_id}'"
            )
            Interface._render_guideline_tool_associations(association.tool_associations)

        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_guideline_tool_association(
        ctx: click.Context,
        agent_id: str,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        try:
            association_id = Actions.remove_guideline_tool_association(
                ctx, agent_id, guideline_id, service_name, tool_name
            )

            Interface._write_success(f"Removed tool association (id={association_id})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_freshness_rules(freshness_rules: FreshnessRules | None) -> str:
        if freshness_rules is None:
            return ""
        parts: list[str] = []
        if freshness_rules.months:
            months = ", ".join(str(m) for m in freshness_rules.months)
            parts.append(f"Months: {months}")
        if freshness_rules.days_of_month:
            days_of_month = ", ".join(str(d) for d in freshness_rules.days_of_month)
            parts.append(f"Days of Month: {days_of_month}")
        if freshness_rules.days_of_week:
            days_of_week = ", ".join(freshness_rules.days_of_week)
            parts.append(f"Days of Week: {days_of_week}")
        if freshness_rules.hours:
            hours = ", ".join(str(h) for h in freshness_rules.hours)
            parts.append(f"Hours: {hours}")
        if freshness_rules.minutes:
            minutes = ", ".join(str(m) for m in freshness_rules.minutes)
            parts.append(f"Minutes: {minutes}")
        if freshness_rules.seconds:
            seconds = ", ".join(str(s) for s in freshness_rules.seconds)
            parts.append(f"Seconds: {seconds}")
        if not parts:
            return "None"
        return "; ".join(parts)

    @staticmethod
    def _render_variable(variable: ContextVariable) -> None:
        Interface._print_table(
            [
                {
                    "ID": variable.id,
                    "Name": variable.name,
                    "Description": variable.description or "",
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
                "ID": variable.id,
                "Name": variable.name,
                "Description": variable.description or "",
                "Service Name": variable.tool_id.service_name if variable.tool_id else "",
                "Tool Name": variable.tool_id.tool_name if variable.tool_id else "",
                "Freshness Rules": Interface._render_freshness_rules(variable.freshness_rules),
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

        Interface._write_success(f"Added variable (id={variable.id})")
        Interface._render_variable(variable)

    @staticmethod
    def delete_variable(ctx: click.Context, agent_id: str, variable_id: str) -> None:
        try:
            Actions.delete_variable(ctx, agent_id, variable_id)
            Interface._write_success(f"Removed variable '{variable_id}'")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_variable_key_value_pairs(
        pairs: dict[str, ContextVariableValue],
    ) -> None:
        values_items: list[dict[str, Any]] = [
            {
                "ID": value.id,
                "Key": key,
                "Value": value.data,
                "Last Modified": reformat_datetime(value.last_modified),
            }
            for key, value in pairs.items()
        ]

        Interface._print_table(values_items)

    @staticmethod
    def set_variable_value(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
        key: str,
        value: str,
    ) -> None:
        try:
            variable = Actions.view_variable(ctx, agent_id, variable_id)
            cv_value = Actions.set_variable_value(
                ctx=ctx,
                agent_id=agent_id,
                variable_id=variable.id,
                key=key,
                value=value,
            )

            Interface._write_success(f"Added value (id={cv_value.id})")
            Interface._render_variable_key_value_pairs({key: cv_value})
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_variable(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
    ) -> None:
        try:
            variable = Actions.view_variable(ctx, agent_id, variable_id)

            read_variable_result = Actions.read_variable(
                ctx,
                agent_id,
                variable.id,
                include_values=True,
            )

            Interface._render_variable(read_variable_result.context_variable)

            if not read_variable_result.key_value_pairs:
                rich.print("No values are available")
                return

            pairs: dict[str, ContextVariableValue] = {}
            for k, v in read_variable_result.key_value_pairs.items():
                if v:
                    pairs[k] = v

            Interface._render_variable_key_value_pairs(pairs)

        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_variable_value(
        ctx: click.Context,
        agent_id: str,
        variable_id: str,
        key: str,
    ) -> None:
        try:
            variable = Actions.view_variable(ctx, agent_id, variable_id)
            value = Actions.read_variable_value(ctx, agent_id, variable.id, key)

            Interface._render_variable_key_value_pairs({key: value})
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def create_service(
        ctx: click.Context,
        name: str,
        kind: str,
        url: str,
        source: str,
    ) -> None:
        try:
            result = Actions.create_service(ctx, name, kind, url, source)

            Interface._write_success(f"Added service '{name}'")
            Interface._print_table([result.dict()])
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_service(
        ctx: click.Context,
        name: str,
    ) -> None:
        try:
            Actions.delete_service(ctx, name)

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

        service_items: list[dict[str, Any]] = [
            {
                "Name": service.name,
                "Type": service.kind,
                "Source": service.url,
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
            service = Actions.view_service(ctx, service_name)
            rich.print(Text("Name:", style="bold"), service.name)
            rich.print(Text("Kind:", style="bold"), service.kind)
            rich.print(Text("Source:", style="bold"), service.url)

            if service.tools:
                rich.print(Text("Tools:", style="bold"))
                for tool in service.tools:
                    rich.print(Text("    Name:", style="bold"), tool.name)
                    if tool.description:
                        rich.print(
                            Text("    Description:\n       ", style="bold"),
                            tool.description,
                        )

                    if tool.parameters:
                        rich.print(Text("    Parameters:", style="bold"))
                        for param_name, param_desc in tool.parameters.items():
                            rich.print(Text(f"      - {param_name}:", style="bold"), end=" ")
                            rich.print(param_desc)

                        rich.print("\n")
            else:
                rich.print("\nNo tools available for this service.")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_customer(customers: list[Customer]) -> None:
        customer_items: list[dict[str, Any]] = [
            {
                "ID": customer.id,
                "Name": customer.name,
                "Extra": customer.extra,
            }
            for customer in customers
        ]

        Interface._print_table(customer_items)

    @staticmethod
    def list_customers(ctx: click.Context) -> None:
        try:
            customers = Actions.list_customers(ctx)
            if not customers:
                rich.print(Text("No customers found", style="bold yellow"))
                return

            Interface._render_customer(customers)
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def create_customer(ctx: click.Context, name: str) -> None:
        try:
            customer = Actions.create_customer(ctx, name)
            Interface._write_success(f"Added customer (id={customer.id})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_customer(ctx: click.Context, customer_id: str) -> None:
        try:
            Actions.delete_customer(ctx, customer_id=customer_id)
            Interface._write_success(f"Removed customer (id={customer_id})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_customer(ctx: click.Context, customer_id: str) -> None:
        try:
            customer = Actions.view_customer(ctx, customer_id)
            Interface._render_customer([customer])
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_customer_extra(ctx: click.Context, customer_id: str, key: str, value: str) -> None:
        try:
            Actions.add_customer_extra(ctx, customer_id, key, value)
            Interface._write_success(
                f"Added extra key '{key}' with value '{value}' to customer {customer_id}"
            )
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_customer_extra(ctx: click.Context, customer_id: str, key: str) -> None:
        try:
            Actions.remove_customer_extra(ctx, customer_id, key)
            Interface._write_success(f"Removed extra key '{key}' from customer {customer_id}")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_customer_tag(ctx: click.Context, customer_id: str, tag_id: str) -> None:
        try:
            Actions.add_customer_tag(ctx, customer_id, tag_id)
            Interface._write_success(f"Added tag '{tag_id}' to customer {customer_id}")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_customer_tag(ctx: click.Context, customer_id: str, tag_id: str) -> None:
        try:
            Actions.remove_customer_tag(ctx, customer_id, tag_id)
            Interface._write_success(f"Removed tag '{tag_id}' from customer {customer_id}")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_tag(tags: list[Tag]) -> None:
        tag_items: list[dict[str, Any]] = [
            {
                "ID": tag.id,
                "Name": tag.name,
            }
            for tag in tags
        ]

        Interface._print_table(tag_items)

    @staticmethod
    def list_tags(ctx: click.Context) -> None:
        try:
            tags = Actions.list_tags(ctx)
            if not tags:
                rich.print("No tags found.")
                return

            Interface._render_tag(tags)
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def create_tag(ctx: click.Context, name: str) -> None:
        try:
            tag = Actions.create_tag(ctx, name=name)
            Interface._write_success(f"Added tag (id={tag.id})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_tag(ctx: click.Context, tag_id: str) -> None:
        try:
            tag = Actions.view_tag(ctx, tag_id=tag_id)
            Interface._render_tag([tag])
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def update_tag(ctx: click.Context, tag_id: str, name: str) -> None:
        try:
            Actions.update_tag(ctx, tag_id=tag_id, name=name)
            Interface._write_success(f"Updated tag (id={tag_id}, name={name})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_tag(ctx: click.Context, tag_id: str) -> None:
        try:
            Actions.delete_tag(ctx, tag_id=tag_id)
            Interface._write_success(f"Removed tag (id={tag_id})")
        except Exception as e:
            Interface._write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)


async def async_main() -> None:
    click_completion.init()  # type: ignore

    @dataclass(frozen=True)
    class Config:
        server_address: str
        client: ParlantClient

    @click.group
    @click.option(
        "-s",
        "--server",
        type=str,
        help="Server address",
        metavar="ADDRESS[:PORT]",
        default="http://localhost:8000",
    )
    @click.pass_context
    def cli(ctx: click.Context, server: str) -> None:
        if not ctx.obj:
            ctx.obj = Config(server_address=server, client=ParlantClient(base_url=server))

    @cli.command(help="Generate shell completion code")
    @click.option("-s", "--shell", type=str, help="Shell program (bash, zsh, etc.)", required=True)
    def complete(shell: str) -> None:
        click.echo(click_completion.get_code(shell))  # type: ignore

    @cli.group(help="Manage agents")
    def agent() -> None:
        pass

    @agent.command("create", help="Create an agent")
    @click.option("--name", type=str, help="Agent name", required=True)
    @click.option("--description", type=str, help="Agent description", required=False)
    @click.option(
        "--max-engine-iterations",
        type=int,
        help="Max engine iterations",
        required=False,
    )
    @click.pass_context
    def agent_create(
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

    @agent.command("delete", help="Delete an agent")
    @click.option("--id", type=str, help="Agent ID", required=True)
    @click.pass_context
    def agent_remove(ctx: click.Context, id: str) -> None:
        Interface.delete_agent(ctx, id)

    @agent.command("view", help="View an agent")
    @click.option("--id", type=str, help="Agent ID", required=True)
    @click.pass_context
    def agent_view(ctx: click.Context, agent_id: str) -> None:
        Interface.view_agent(ctx, agent_id)

    @agent.command("list", help="List agents")
    @click.pass_context
    def agent_list(ctx: click.Context) -> None:
        Interface.list_agents(ctx)

    @agent.command("update", help="Update an agent's details")
    @click.option(
        "--id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option(
        "--name",
        type=str,
        help="Agent Name",
        required=False,
    )
    @click.option("--description", type=str, help="Agent description", required=False)
    @click.option(
        "--max-engine-iterations",
        type=int,
        help="Max engine iterations",
        required=False,
    )
    @click.pass_context
    def agent_update(
        ctx: click.Context,
        id: str,
        name: Optional[str],
        description: Optional[str],
        max_engine_iterations: Optional[int],
    ) -> None:
        id = id if id else Interface.get_default_agent(ctx)
        assert id

        Interface.update_agent(ctx, id, name, description, max_engine_iterations)

    @cli.group(help="Manage sessions")
    def session() -> None:
        pass

    @session.command("create", help="Create a session")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option(
        "--customer-id",
        type=str,
        help="Customer ID (defaults to the guest customer)",
        metavar="ID",
        required=False,
    )
    @click.option("--title", type=str, help="Session Title", metavar="TITLE", required=False)
    @click.pass_context
    def session_create(
        ctx: click.Context,
        agent_id: str,
        customer_id: Optional[str],
        title: Optional[str],
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.create_session(ctx, agent_id, customer_id, title)

    @session.command("delete", help="Delete a session")
    @click.option("--id", type=str, metavar="ID", help="Session ID", required=True)
    @click.pass_context
    def session_delete(
        ctx: click.Context,
        id: str,
    ) -> None:
        Interface.delete_session(ctx, id)

    @session.command("update", help="Update a session")
    @click.option("--title", type=str, help="Session Title", metavar="TITLE", required=False)
    @click.option("--id", type=str, metavar="ID", help="Session ID", required=True)
    @click.pass_context
    def session_update(
        ctx: click.Context,
        id: str,
        title: Optional[str],
    ) -> None:
        Interface.update_session(ctx, id, title, None)

    @session.command("list", help="List sessions")
    @click.option(
        "--agent-id",
        type=str,
        help="Filter by agent ID",
        metavar="ID",
        required=False,
    )
    @click.option(
        "--customer-id",
        type=str,
        help="Filter by Customer ID",
        metavar="ID",
        required=False,
    )
    @click.pass_context
    def session_list(
        ctx: click.Context, agent_id: Optional[str], customer_id: Optional[str]
    ) -> None:
        Interface.list_sessions(ctx, agent_id, customer_id)

    @session.command("view", help="View session content")
    @click.argument("session_id")
    @click.pass_context
    def session_view(ctx: click.Context, session_id: str) -> None:
        Interface.view_session(ctx, session_id)

    @session.command("inspect", help="Inspect an event from a session")
    @click.option("--session-id", type=str, help="Session ID", metavar="ID", required=True)
    @click.option("--event-id", type=str, help="Event ID", metavar="ID", required=True)
    @click.pass_context
    def session_inspect(ctx: click.Context, session_id: str, event_id: str) -> None:
        Interface.inspect_event(ctx, session_id, event_id)

    @cli.group(help="Manage an agent's glossary")
    def glossary() -> None:
        pass

    @glossary.command("create", help="Create a term")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--name", type=str, help="Term name", required=True)
    @click.option("--description", type=str, help="Term description", required=True)
    @click.option(
        "--synonyms",
        type=str,
        help="Comma-separated list of synonyms",
        metavar="LIST",
        required=False,
    )
    @click.pass_context
    def glossary_create(
        ctx: click.Context,
        agent_id: str,
        name: str,
        description: str,
        synonyms: Optional[str],
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.create_term(
            ctx,
            agent_id,
            name,
            description,
            (synonyms or "").split(","),
        )

    @glossary.command("update", help="Update a term")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--id", type=str, help="Term ID", metavar="ID", required=True)
    @click.option(
        "--name",
        type=str,
        help="Term name",
        metavar="NAME",
        required=False,
    )
    @click.option(
        "--description",
        type=str,
        help="Term description",
        required=False,
    )
    @click.option(
        "--synonyms",
        type=str,
        help="Comma-separated list of synonyms",
        metavar="LIST",
        required=False,
    )
    @click.pass_context
    def glossary_update(
        ctx: click.Context,
        agent_id: str,
        id: str,
        name: Optional[str],
        description: Optional[str],
        synonyms: Optional[str],
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.update_term(
            ctx,
            agent_id,
            id,
            name,
            description,
            (synonyms or "").split(","),
        )

    @glossary.command("delete", help="Delete a term")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--id", type=str, metavar="ID", help="Term ID", required=True)
    @click.pass_context
    def glossary_delete(
        ctx: click.Context,
        agent_id: str,
        id: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.delete_term(ctx, agent_id, id)

    @glossary.command("list", help="List terms")
    @click.option(
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

    @guideline.command("create", help="Create a guideline")
    @click.option(
        "--condition",
        type=str,
        help="A statement describing when the guideline should apply",
        required=True,
    )
    @click.option(
        "--action",
        type=str,
        help="The instruction to perform when the guideline applies",
        required=True,
    )
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option(
        "--check/--no-check",
        type=bool,
        show_default=True,
        default=True,
        help="Check for contradictions with existing guidelines",
    )
    @click.option(
        "--connect/--no-connect",
        type=bool,
        show_default=True,
        default=True,
        help="Connect this guideline with causually related guidelines",
    )
    @click.pass_context
    def guideline_create(
        ctx: click.Context,
        agent_id: str,
        condition: str,
        action: str,
        check: bool,
        index: bool,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.create_guideline(
            ctx=ctx,
            agent_id=agent_id,
            condition=condition,
            action=action,
            check=check,
            index=index,
        )

    @guideline.command("update", help="Update a guideline")
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.option(
        "--condition",
        type=str,
        help="A statement describing when the guideline should apply",
        required=True,
    )
    @click.option(
        "--action",
        type=str,
        help="The instruction to perform when the guideline applies",
        required=True,
    )
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option(
        "--check/--no-check",
        type=bool,
        show_default=True,
        default=True,
        help="Check for contradictions with existing guidelines",
    )
    @click.option(
        "--connect/--no-connect",
        type=bool,
        show_default=True,
        default=True,
        help="Connect this guideline with causually related guidelines",
    )
    @click.pass_context
    def guideline_update(
        ctx: click.Context,
        agent_id: str,
        id: str,
        condition: str,
        action: str,
        check: bool,
        index: bool,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.update_guideline(
            ctx=ctx,
            agent_id=agent_id,
            condition=condition,
            action=action,
            guideline_id=id,
            check=check,
            index=index,
        )

    @guideline.command("delete", help="Delete a guideline")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.pass_context
    def guideline_delete(
        ctx: click.Context,
        agent_id: str,
        id: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.delete_guideline(
            ctx=ctx,
            agent_id=agent_id,
            guideline_id=id,
        )

    @guideline.command("view", help="View a guideline")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.pass_context
    def guideline_view(
        ctx: click.Context,
        agent_id: str,
        id: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.view_guideline(
            ctx=ctx,
            agent_id=agent_id,
            guideline_id=id,
        )

    @guideline.command("list", help="List guidelines")
    @click.option(
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
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option(
        "--suggestive",
        is_flag=True,
        show_default=True,
        default=False,
        help="Make the entailment suggestive rather than definite",
    )
    @click.option("--source", type=str, metavar="ID", help="Source guideline ID", required=True)
    @click.option("--target", type=str, metavar="ID", help="Target guideline ID", required=True)
    @click.pass_context
    def guideline_entail(
        ctx: click.Context,
        agent_id: str,
        suggestive: bool,
        source: str,
        target: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.create_entailment(
            ctx=ctx,
            agent_id=agent_id,
            source_guideline_id=source,
            target_guideline_id=target,
            kind="suggests" if suggestive else "entails",
        )

    @guideline.command("disentail", help="Delete an entailment between two guidelines")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--source", type=str, metavar="ID", help="Source guideline ID", required=True)
    @click.option("--target", type=str, metavar="ID", help="Target guideline ID", required=True)
    @click.pass_context
    def guideline_disentail(
        ctx: click.Context,
        agent_id: str,
        source: str,
        target: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.remove_entailment(
            ctx=ctx,
            agent_id=agent_id,
            source_guideline_id=source,
            target_guideline_id=target,
        )

    @guideline.command("tool-enable", help="Allow a guideline to make use of a tool")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.option(
        "--service-name",
        type=str,
        metavar="NAME",
        help="The name of the service containing the tool",
        required=True,
    )
    @click.option("--tool-name", type=str, metavar="NAME", help="Tool name", required=True)
    @click.pass_context
    def guideline_enable_tool(
        ctx: click.Context,
        agent_id: Optional[str],
        id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.add_guideline_tool_association(
            ctx=ctx,
            agent_id=agent_id,
            guideline_id=id,
            service_name=service_name,
            tool_name=tool_name,
        )

    @guideline.command("tool-disable", help="Disallow a guideline to make use of a tool")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.option(
        "--service-name",
        type=str,
        metavar="NAME",
        help="The name of the service containing the tool",
        required=True,
    )
    @click.option("--tool-name", type=str, metavar="NAME", help="Tool name", required=True)
    @click.pass_context
    def guideline_disable_tool(
        ctx: click.Context,
        agent_id: Optional[str],
        id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.remove_guideline_tool_association(
            ctx=ctx,
            agent_id=agent_id,
            guideline_id=id,
            service_name=service_name,
            tool_name=tool_name,
        )

    @cli.group(help="Manage an agent's context variables")
    def variable() -> None:
        pass

    @variable.command("list", help="List variables")
    @click.option(
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

    @variable.command("create", help="Create a context variable")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--description", type=str, help="Variable description", required=False)
    @click.option("--name", type=str, metavar="NAME", help="Variable name", required=True)
    @click.pass_context
    def variable_create(
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

    @variable.command("delete", help="Delete a context variable")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @click.pass_context
    def variable_delete(
        ctx: click.Context,
        agent_id: Optional[str],
        id: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.delete_variable(
            ctx=ctx,
            agent_id=agent_id,
            variable_id=id,
        )

    @variable.command("set", help="Set the value of a key under a context variable")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @click.option("--key", type=str, metavar="NAME", help="The key (e.g. customer ID or tag)")
    @click.option("--value", type=str, metavar="TEXT", help="The key's value")
    @click.pass_context
    def variable_set(
        ctx: click.Context,
        agent_id: Optional[str],
        id: str,
        key: str,
        value: str,
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        Interface.set_variable_value(
            ctx=ctx,
            agent_id=agent_id,
            variable_id=id,
            key=key,
            value=value,
        )

    @variable.command("get", help="Get the value(s) of a variable")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID (defaults to the first agent)",
        metavar="ID",
        required=False,
    )
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @click.option("--key", type=str, metavar="NAME", help="The key (e.g. customer ID or tag)")
    @click.pass_context
    def variable_get(
        ctx: click.Context,
        agent_id: Optional[str],
        id: str,
        key: Optional[str],
    ) -> None:
        agent_id = agent_id if agent_id else Interface.get_default_agent(ctx)
        assert agent_id

        if key:
            Interface.view_variable_value(
                ctx=ctx,
                agent_id=agent_id,
                variable_id=id,
                key=key,
            )
        else:
            Interface.view_variable(
                ctx=ctx,
                agent_id=agent_id,
                variable_id=id,
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
        Interface.create_service(ctx, name, kind, url, source)

    @service.command("remove", help="Remove a service")
    @click.argument("name", type=str)
    @click.pass_context
    def service_remove(ctx: click.Context, name: str) -> None:
        Interface.delete_service(ctx, name)

    @service.command("list", help="List all services")
    @click.pass_context
    def service_list(ctx: click.Context) -> None:
        Interface.list_services(ctx)

    @service.command("view", help="View a specific service and its tools")
    @click.argument("name", type=str)
    @click.pass_context
    def service_view(ctx: click.Context, name: str) -> None:
        Interface.view_service(ctx, name)

    @cli.group(help="Manage customers")
    def customer() -> None:
        pass

    @customer.command("list", help="List all customers")
    @click.pass_context
    def list_customers(ctx: click.Context) -> None:
        Interface.list_customers(ctx)

    @customer.command("add", help="Add a new customer")
    @click.argument("name")
    @click.pass_context
    def add_customer(ctx: click.Context, name: str) -> None:
        Interface.create_customer(ctx, name)

    @customer.command("remove", help="Remove a customer")
    @click.argument("customer_id", type=str)
    @click.pass_context
    def remove_customer(ctx: click.Context, customer_id: str) -> None:
        Interface.delete_customer(ctx, customer_id)

    @customer.command("view", help="View a customer's details")
    @click.argument("customer_id", type=str)
    @click.pass_context
    def view_customer(ctx: click.Context, customer_id: str) -> None:
        Interface.view_customer(ctx, customer_id)

    @customer.command("add-extra", help="Add extra information to a customer")
    @click.argument("customer_id", type=str)
    @click.argument("key")
    @click.argument("value")
    @click.pass_context
    def add_customer_extra(ctx: click.Context, customer_id: str, key: str, value: str) -> None:
        Interface.add_customer_extra(ctx, customer_id, key, value)

    @customer.command("remove-extra", help="Remove extra information from a customer")
    @click.argument("customer_id", type=str)
    @click.argument("key")
    @click.pass_context
    def remove_customer_extra(ctx: click.Context, customer_id: str, key: str) -> None:
        Interface.remove_customer_extra(ctx, customer_id, key)

    @customer.command("add-tag", help="Add a tag to a customer")
    @click.argument("customer_id", type=str)
    @click.argument("tag_id", type=str)
    @click.pass_context
    def add_customer_tag(ctx: click.Context, customer_id: str, tag_id: str) -> None:
        Interface.add_customer_tag(ctx, customer_id, tag_id)

    @customer.command("remove-tag", help="Remove a tag from a customer")
    @click.argument("customer_id", type=str)
    @click.argument("tag_id", type=str)
    @click.pass_context
    def remove_customer_tag(ctx: click.Context, customer_id: str, tag_id: str) -> None:
        Interface.remove_customer_tag(ctx, customer_id, tag_id)

    @cli.group(help="Manage tags")
    def tag() -> None:
        """Group of commands to manage tags."""

    @tag.command("list", help="List all tags")
    @click.pass_context
    def list_tags(ctx: click.Context) -> None:
        Interface.list_tags(ctx)

    @tag.command("add", help="Add a new tag")
    @click.argument("name", type=str)
    @click.pass_context
    def add_tag(ctx: click.Context, name: str) -> None:
        Interface.create_tag(ctx, name)

    @tag.command("view", help="View details of a tag")
    @click.argument("tag_id", type=str)
    @click.pass_context
    def view_tag(ctx: click.Context, tag_id: str) -> None:
        Interface.view_tag(ctx, tag_id)

    @tag.command("update", help="Update a tag's details")
    @click.argument("tag_id", type=str)
    @click.argument("name", type=str)
    @click.pass_context
    def update_tag(ctx: click.Context, tag_id: str, name: str) -> None:
        Interface.update_tag(ctx, tag_id, name)

    @tag.command("remove", help="Remove a tag")
    @click.argument("tag_id", type=str)
    @click.pass_context
    def remove_tag(ctx: click.Context, tag_id: str) -> None:
        Interface.delete_tag(ctx, tag_id)

    cli()


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
    sys.exit(get_exit_status())
