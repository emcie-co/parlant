# Copyright 2024 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code=import-untyped

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlparse
import click
import click.shell_completion
import click_completion  # type: ignore
from dataclasses import dataclass
from datetime import datetime
import requests
import rich
from rich import box
from rich.table import Table
from rich.text import Text
import sys
from typing import Any, Iterator, Optional, cast

from parlant.client import ParlantClient
from parlant.client.core import ApiError
from parlant.client.types import (
    Agent,
    AgentTagUpdateParams,
    ContextVariable,
    ContextVariableReadResult,
    ContextVariableValue,
    ContextVariableTagsUpdateParams,
    Event,
    EventInspectionResult,
    Guideline,
    GuidelineConnection,
    GuidelineConnectionAddition,
    GuidelineConnectionUpdateParams,
    GuidelineToolAssociation,
    GuidelineToolAssociationUpdateParams,
    GuidelineTagsUpdateParams,
    GuidelineWithConnectionsAndToolAssociations,
    OpenApiServiceParams,
    SdkServiceParams,
    Service,
    Session,
    Term,
    TermTagsUpdateParams,
    ToolId,
    Customer,
    CustomerExtraUpdateParams,
    CustomerTagUpdateParams,
    Tag,
    ConsumptionOffsetsUpdateParams,
    Utterance,
    UtteranceField,
)
from websocket import WebSocketConnectionClosedException, create_connection

INDENT = "  "


class FastExit(Exception):
    pass


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
    def _fetch_tag_id(
        ctx: click.Context,
        tag: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        tags = client.tags.list()
        for t in tags:
            if t.name == tag or t.id == tag:
                return t.id

        raise Exception(f"Tag ({tag}) not found")

    @staticmethod
    def create_agent(
        ctx: click.Context,
        name: str,
        description: Optional[str],
        max_engine_iterations: Optional[int],
        composition_mode: Optional[str],
        tags: list[str],
    ) -> Agent:
        client = cast(ParlantClient, ctx.obj.client)

        return client.agents.create(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
            composition_mode=composition_mode,
            tags=list(set([Actions._fetch_tag_id(ctx, t) for t in tags])),
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
        composition_mode: Optional[str] = None,
    ) -> Agent:
        client = cast(ParlantClient, ctx.obj.client)

        return client.agents.update(
            agent_id,
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
            composition_mode=composition_mode,
        )

    @staticmethod
    def add_tag(
        ctx: click.Context,
        agent_id: str,
        tag: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.agents.update(
            agent_id=agent_id,
            tags=AgentTagUpdateParams(add=[Actions._fetch_tag_id(ctx, tag)]),
        )

    @staticmethod
    def remove_tag(
        ctx: click.Context,
        agent_id: str,
        tag: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.agents.update(
            agent_id=agent_id,
            tags=AgentTagUpdateParams(remove=[Actions._fetch_tag_id(ctx, tag)]),
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
        return client.sessions.list_events(session_id=session_id, wait_for_data=0)

    @staticmethod
    def create_term(
        ctx: click.Context,
        name: str,
        description: str,
        synonyms: list[str],
        tags: list[str],
    ) -> Term:
        client = cast(ParlantClient, ctx.obj.client)

        return client.glossary.create_term(
            name=name,
            description=description,
            synonyms=synonyms,
            tags=list(set([Actions._fetch_tag_id(ctx, t) for t in tags])),
        )

    @staticmethod
    def update_term(
        ctx: click.Context,
        term_id: str,
        name: Optional[str],
        description: Optional[str],
        synonyms: list[str],
    ) -> Term:
        client = cast(ParlantClient, ctx.obj.client)

        return client.glossary.update_term(
            term_id,
            name=name,
            description=description,
            synonyms=synonyms,
        )

    @staticmethod
    def delete_term(
        ctx: click.Context,
        term_id: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.glossary.delete_term(term_id)

    @staticmethod
    def list_terms(
        ctx: click.Context,
        tag: Optional[str] = None,
    ) -> list[Term]:
        client = cast(ParlantClient, ctx.obj.client)
        if tag:
            return client.glossary.list_terms(tag_id=Actions._fetch_tag_id(ctx, tag))
        else:
            return client.glossary.list_terms()

    @staticmethod
    def add_term_tag(
        ctx: click.Context,
        term_id: str,
        tag: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.glossary.update_term(
            term_id, tags=TermTagsUpdateParams(add=[Actions._fetch_tag_id(ctx, tag)])
        )

    @staticmethod
    def remove_term_tag(
        ctx: click.Context,
        term_id: str,
        tag: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.glossary.update_term(
            term_id, tags=TermTagsUpdateParams(remove=[Actions._fetch_tag_id(ctx, tag)])
        )

    @staticmethod
    def create_guideline(
        ctx: click.Context,
        condition: str,
        action: str,
        tags: list[str],
    ) -> Guideline:
        client = cast(ParlantClient, ctx.obj.client)

        return client.guidelines.create(
            condition=condition,
            action=action,
            tags=list(set([Actions._fetch_tag_id(ctx, t) for t in tags])),
        )

    @staticmethod
    def update_guideline(
        ctx: click.Context,
        guideline_id: str,
        condition: Optional[str] = None,
        action: Optional[str] = None,
    ) -> GuidelineWithConnectionsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)

        return client.guidelines.update(guideline_id, condition=condition, action=action)

    @staticmethod
    def delete_guideline(
        ctx: click.Context,
        guideline_id: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.guidelines.delete(guideline_id)

    @staticmethod
    def view_guideline(
        ctx: click.Context,
        guideline_id: str,
    ) -> GuidelineWithConnectionsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)
        return client.guidelines.retrieve(guideline_id)

    @staticmethod
    def list_guidelines(
        ctx: click.Context,
        tag: Optional[str],
    ) -> list[Guideline]:
        client = cast(ParlantClient, ctx.obj.client)
        if tag:
            return client.guidelines.list(tag_id=Actions._fetch_tag_id(ctx, tag))
        else:
            return client.guidelines.list()

    @staticmethod
    def create_entailment(
        ctx: click.Context,
        source_guideline_id: str,
        target_guideline_id: str,
    ) -> GuidelineWithConnectionsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)

        return client.guidelines.update(
            source_guideline_id,
            connections=GuidelineConnectionUpdateParams(
                add=[
                    GuidelineConnectionAddition(
                        source=source_guideline_id,
                        target=target_guideline_id,
                    ),
                ]
            ),
        )

    @staticmethod
    def remove_entailment(
        ctx: click.Context,
        source_guideline_id: str,
        target_guideline_id: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        guideline_result = client.guidelines.retrieve(source_guideline_id)
        connections = guideline_result.connections

        if connection := next(
            (c for c in connections if target_guideline_id in [c.source.id, c.target.id]),
            None,
        ):
            client.guidelines.update(
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
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> GuidelineWithConnectionsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)

        return client.guidelines.update(
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
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        guideline_result = client.guidelines.retrieve(guideline_id)
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
    def enable_guideline(
        ctx: click.Context,
        guideline_ids: tuple[str],
    ) -> list[Guideline]:
        client = cast(ParlantClient, ctx.obj.client)

        return [
            client.guidelines.update(guideline_id, enabled=True).guideline
            for guideline_id in guideline_ids
        ]

    @staticmethod
    def disable_guideline(
        ctx: click.Context,
        guideline_ids: tuple[str],
    ) -> list[Guideline]:
        client = cast(ParlantClient, ctx.obj.client)

        return [
            client.guidelines.update(guideline_id, enabled=False).guideline
            for guideline_id in guideline_ids
        ]

    @staticmethod
    def add_guideline_tag(
        ctx: click.Context,
        guideline_id: str,
        tag: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.guidelines.update(
            guideline_id, tags=GuidelineTagsUpdateParams(add=[Actions._fetch_tag_id(ctx, tag)])
        )

    @staticmethod
    def remove_guideline_tag(
        ctx: click.Context,
        guideline_id: str,
        tag: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.guidelines.update(
            guideline_id, tags=GuidelineTagsUpdateParams(remove=[Actions._fetch_tag_id(ctx, tag)])
        )

    @staticmethod
    def list_variables(
        ctx: click.Context,
        tag: Optional[str],
    ) -> list[ContextVariable]:
        client = cast(ParlantClient, ctx.obj.client)
        if tag:
            return client.context_variables.list(tag_id=Actions._fetch_tag_id(ctx, tag))
        else:
            return client.context_variables.list()

    @staticmethod
    def create_variable(
        ctx: click.Context,
        name: str,
        description: str,
        service_name: Optional[str],
        tool_name: Optional[str],
        freshness_rules: Optional[str],
        tags: list[str],
    ) -> ContextVariable:
        client = cast(ParlantClient, ctx.obj.client)

        return client.context_variables.create(
            name=name,
            description=description,
            tool_id=ToolId(service_name=service_name, tool_name=tool_name)
            if service_name and tool_name
            else None,
            freshness_rules=freshness_rules,
            tags=list(set([Actions._fetch_tag_id(ctx, t) for t in tags])),
        )

    @staticmethod
    def update_variable(
        ctx: click.Context,
        variable_id: str,
        name: Optional[str],
        description: Optional[str],
        service_name: Optional[str],
        tool_name: Optional[str],
        freshness_rules: Optional[str],
    ) -> ContextVariable:
        client = cast(ParlantClient, ctx.obj.client)

        return client.context_variables.update(
            variable_id,
            name=name,
            description=description,
            tool_id=ToolId(service_name=service_name, tool_name=tool_name)
            if service_name and tool_name
            else None,
            freshness_rules=freshness_rules,
        )

    @staticmethod
    def delete_variable(
        ctx: click.Context,
        variable_id: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.context_variables.delete(variable_id)

    @staticmethod
    def set_variable_value(
        ctx: click.Context,
        variable_id: str,
        key: str,
        value: str,
    ) -> ContextVariableValue:
        client = cast(ParlantClient, ctx.obj.client)

        return client.context_variables.set_value(
            variable_id,
            key,
            data=value,
        )

    @staticmethod
    def view_variable(
        ctx: click.Context,
        variable_id: str,
        include_values: bool,
    ) -> ContextVariableReadResult:
        client = cast(ParlantClient, ctx.obj.client)

        return client.context_variables.retrieve(
            variable_id,
            include_values=include_values,
        )

    @staticmethod
    def view_variable_value(
        ctx: click.Context,
        variable_id: str,
        key: str,
    ) -> ContextVariableValue:
        client = cast(ParlantClient, ctx.obj.client)

        return client.context_variables.get_value(
            variable_id,
            key,
        )

    @staticmethod
    def delete_variable_value(
        ctx: click.Context,
        variable_id: str,
        key: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.context_variables.delete_value(variable_id, key)

    @staticmethod
    def add_variable_tag(
        ctx: click.Context,
        variable_id: str,
        tag: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.context_variables.update(
            variable_id, tags=ContextVariableTagsUpdateParams(add=[Actions._fetch_tag_id(ctx, tag)])
        )

    @staticmethod
    def remove_variable_tag(
        ctx: click.Context,
        variable_id: str,
        tag: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.context_variables.update(
            variable_id,
            tags=ContextVariableTagsUpdateParams(remove=[Actions._fetch_tag_id(ctx, tag)]),
        )

    @staticmethod
    def create_or_update_service(
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
        tags: list[str],
    ) -> Customer:
        client = cast(ParlantClient, ctx.obj.client)
        return client.customers.create(
            name=name,
            extra={},
            tags=list(set([Actions._fetch_tag_id(ctx, t) for t in tags])),
        )

    @staticmethod
    def update_customer(
        ctx: click.Context,
        customer_id: str,
        name: str,
    ) -> Customer:
        client = cast(ParlantClient, ctx.obj.client)
        return client.customers.update(customer_id=customer_id, name=name)

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
    def add_customer_tag(
        ctx: click.Context,
        customer_id: str,
        tag: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.customers.update(
            customer_id=customer_id,
            tags=CustomerTagUpdateParams(add=[Actions._fetch_tag_id(ctx, tag)]),
        )

    @staticmethod
    def remove_customer_tag(
        ctx: click.Context,
        customer_id: str,
        tag: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.customers.update(
            customer_id=customer_id,
            tags=CustomerTagUpdateParams(remove=[Actions._fetch_tag_id(ctx, tag)]),
        )

    @staticmethod
    def list_tags(ctx: click.Context) -> list[Tag]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.tags.list()

    @staticmethod
    def create_tag(
        ctx: click.Context,
        name: str,
    ) -> Tag:
        client = cast(ParlantClient, ctx.obj.client)
        return client.tags.create(name=name)

    @staticmethod
    def view_tag(
        ctx: click.Context,
        tag: str,
    ) -> Tag:
        client = cast(ParlantClient, ctx.obj.client)
        return client.tags.retrieve(tag_id=Actions._fetch_tag_id(ctx, tag))

    @staticmethod
    def update_tag(
        ctx: click.Context,
        tag: str,
        name: str,
    ) -> Tag:
        client = cast(ParlantClient, ctx.obj.client)
        return client.tags.update(tag_id=Actions._fetch_tag_id(ctx, tag), name=name)

    @staticmethod
    def delete_tag(
        ctx: click.Context,
        tag: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.tags.delete(tag_id=Actions._fetch_tag_id(ctx, tag))

    @staticmethod
    def list_utterances(ctx: click.Context) -> list[Utterance]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.utterances.list()

    @staticmethod
    def view_utterance(ctx: click.Context, utterance_id: str) -> Utterance:
        client = cast(ParlantClient, ctx.obj.client)
        return client.utterances.retrieve(utterance_id=utterance_id)

    @staticmethod
    def load_utterances(ctx: click.Context, path: Path) -> list[Utterance]:
        with open(path, "r") as file:
            data = json.load(file)

        client = cast(ParlantClient, ctx.obj.client)

        for utterance in client.utterances.list():
            client.utterances.delete(utterance_id=utterance.id)

        utterances = []
        tag_ids = {tag.name: tag.id for tag in client.tags.list()}

        for utterance_data in data.get("utterances", []):
            value = utterance_data["value"]
            assert value

            fields = [
                UtteranceField(**utterance_field)
                for utterance_field in utterance_data.get("fields", [])
            ]

            tag_names = utterance_data.get("tags", [])

            utterance = client.utterances.create(
                value=value,
                fields=fields,
                tags=[tag_ids[tag_name] for tag_name in tag_names if tag_name in tag_ids] or None,
            )

            utterances.append(utterance)

        return utterances

    @staticmethod
    def stream_logs(
        ctx: click.Context,
        union_patterns: list[str],
        intersection_patterns: list[str],
    ) -> Iterator[dict[str, Any]]:
        url = f"{ctx.obj.server_address.replace('http', 'ws')}/logs"
        ws = create_connection(url)

        try:
            rich.print(Text("Streaming logs...", style="bold yellow"))

            while True:
                raw_message = ws.recv()
                message = json.loads(raw_message)

                if Actions._log_entry_matches(message, union_patterns, intersection_patterns):
                    yield message
        except KeyboardInterrupt:
            rich.print(Text("Log streaming interrupted by user.", style="bold red"))
        except WebSocketConnectionClosedException:
            Interface.write_error("The WebSocket connection was closed.")
        finally:
            ws.close()

    @staticmethod
    def _log_entry_matches(
        log_entry: dict[str, Any], union_patterns: list[str], intersection_patterns: list[str]
    ) -> bool:
        message = log_entry.get("message", "")

        if not union_patterns and not intersection_patterns:
            return True

        if not union_patterns:
            return all(p in message for p in intersection_patterns)

        if not intersection_patterns:
            return any(p in message for p in union_patterns)

        return any(p in message for p in union_patterns) and all(
            p in message for p in intersection_patterns
        )


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
        ) + (f": {response.json()['detail']}" if "detail" in response.json() else "")
    elif 500 <= response.status_code < 600:
        http_error_msg = (
            f"{response.status_code} Server Error: {reason} for url: {response.url}"
            + (f": {response.json()['detail']}" if "detail" in response.json() else "")
        )

    if http_error_msg:
        raise requests.HTTPError(http_error_msg, response=response)


class Interface:
    @staticmethod
    def _write_success(message: str) -> None:
        rich.print(Text(message, style="bold green"))

    @staticmethod
    def write_error(message: str) -> None:
        rich.print(Text(message, style="bold red"), file=sys.stderr)

    @staticmethod
    def _print_table(data: list[dict[str, Any]]) -> None:
        table = Table(box=box.ROUNDED, border_style="bright_green")

        table.add_column("#", header_style="bright_green", overflow="fold")

        headers = list(data[0].keys()) if data else []

        for header in headers:
            table.add_column(header, header_style="bright_green", overflow="fold")

        for idx, row in enumerate(data, start=1):
            row_values = [str(row.get(h, "")) for h in headers]
            table.add_row(str(idx), *row_values)

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
                "Composition Mode": a.composition_mode.replace("_", "-"),
                "Tags": ", ".join(a.tags or []),
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
        composition_mode: Optional[str],
        tags: list[str],
    ) -> None:
        try:
            agent = Actions.create_agent(
                ctx,
                name,
                description,
                max_engine_iterations,
                composition_mode,
                tags,
            )

            Interface._write_success(f"Added agent (id: {agent.id})")
            Interface._render_agents([agent])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_agent(ctx: click.Context, agent_id: str) -> None:
        try:
            Actions.delete_agent(ctx, agent_id=agent_id)
            Interface._write_success(f"Removed agent (id: {agent_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_agent(ctx: click.Context, agent_id: str) -> None:
        try:
            agent = Actions.view_agent(ctx, agent_id)

            Interface._render_agents([agent])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def list_agents(ctx: click.Context) -> None:
        agents = Actions.list_agents(ctx)

        if not agents:
            rich.print(Text("No data available", style="bold yellow"))
            return

        Interface._render_agents(agents)

    @staticmethod
    def get_default_agent(ctx: click.Context) -> str:
        agents = Actions.list_agents(ctx)

        if not agents:
            Interface.write_error("Error: No agents exist. Please create at least one agent.")
            set_exit_status(1)
            raise FastExit()

        if len(agents) != 1:
            Interface.write_error("Error: There's more than one agent. Please specify --agent-id.")
            set_exit_status(1)
            raise FastExit()

        return str(agents[0].id)

    @staticmethod
    def update_agent(
        ctx: click.Context,
        agent_id: str,
        name: Optional[str],
        description: Optional[str],
        max_engine_iterations: Optional[int],
        composition_mode: Optional[str],
    ) -> None:
        try:
            agent = Actions.update_agent(
                ctx, agent_id, name, description, max_engine_iterations, composition_mode
            )
            Interface._write_success(f"Updated agent (id: {agent_id})")
            Interface._render_agents([agent])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_tag(ctx: click.Context, agent_id: str, tag: str) -> None:
        try:
            Actions.add_tag(ctx, agent_id, tag)
            Interface._write_success(f"Tagged agent (id: {agent_id}, tag: {tag})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_tag(ctx: click.Context, agent_id: str, tag: str) -> None:
        try:
            Actions.remove_tag(ctx, agent_id, tag)
            Interface._write_success(f"Untagged agent (id: {agent_id}, tag: {tag})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_sessions(sessions: list[Session]) -> None:
        session_items = [
            {
                "ID": s.id,
                "Title": s.title or "",
                "Agent ID": s.agent_id,
                "Customer ID": s.customer_id,
                "Creation Date": reformat_datetime(s.creation_utc),
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
                "Deleted": e.deleted,
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
            rich.print(Text("No data available", style="bold yellow"))
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
            rich.print(Text("No data available", style="bold yellow"))
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
        Interface._write_success(f"Added session (id: {session.id})")
        Interface._render_sessions([session])

    @staticmethod
    def delete_session(ctx: click.Context, session_id: str) -> None:
        try:
            Actions.delete_session(ctx, session_id=session_id)
            Interface._write_success(f"Removed session (id: {session_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def update_session(
        ctx: click.Context,
        session_id: str,
        title: Optional[str] = None,
        consumption_offsets: Optional[int] = None,
    ) -> None:
        Actions.update_session(ctx, session_id, consumption_offsets, title)
        Interface._write_success(f"Updated session (id: {session_id})")

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

            rich.print(Text(f"{INDENT}Guideline Matches:", style="bold"))

            if iteration.guideline_matches:
                for match in iteration.guideline_matches:
                    rich.print(f"{INDENT*2}Condition: {match.condition}")
                    rich.print(f"{INDENT*2}Action: {match.action}")
                    rich.print(f"{INDENT*2}Relevance Score: {match.score}/10")
                    rich.print(f"{INDENT*2}Rationale: {match.rationale}\n")
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
    def _render_glossary(terms: list[Term]) -> None:
        term_items: list[dict[str, Any]] = [
            {
                "ID": term.id,
                "Name": term.name,
                "Description": term.description,
                "Synonyms": ", ".join(term.synonyms or []),
                "Tags": ", ".join(term.tags),
            }
            for term in terms
        ]

        Interface._print_table(term_items)

    @staticmethod
    def create_term(
        ctx: click.Context,
        name: str,
        description: str,
        synonyms: list[str],
        tags: list[str],
    ) -> None:
        term = Actions.create_term(
            ctx,
            name,
            description,
            synonyms,
            tags=tags,
        )

        Interface._write_success(f"Added term (id: {term.id})")
        Interface._render_glossary([term])

    @staticmethod
    def update_term(
        ctx: click.Context,
        term_id: str,
        name: Optional[str],
        description: Optional[str],
        synonyms: list[str],
    ) -> None:
        if not name and not description and not synonyms:
            Interface.write_error(
                "Error: No updates provided. Please provide at least one of the following: name, description, or synonyms to update the term."
            )
            return

        term = Actions.update_term(
            ctx,
            term_id,
            name,
            description,
            synonyms,
        )
        Interface._write_success(f"Updated term (id: {term.id})")
        Interface._print_table([term.__dict__])

    @staticmethod
    def delete_term(
        ctx: click.Context,
        term_id: str,
    ) -> None:
        Actions.delete_term(ctx, term_id)

        Interface._write_success(f"Removed term (id: {term_id})")

    @staticmethod
    def list_terms(
        ctx: click.Context,
        tag: Optional[str],
    ) -> None:
        terms = Actions.list_terms(ctx, tag)

        if not terms:
            rich.print(Text("No data available", style="bold yellow"))
            return

        Interface._render_glossary(terms)

    @staticmethod
    def add_term_tag(
        ctx: click.Context,
        term_id: str,
        tag: str,
    ) -> None:
        Actions.add_term_tag(ctx, term_id, tag)

        Interface._write_success(f"Added tag ( {tag}) to term (id: {term_id})")

    @staticmethod
    def remove_term_tag(
        ctx: click.Context,
        term_id: str,
        tag: str,
    ) -> None:
        Actions.remove_term_tag(ctx, term_id, tag)

        Interface._write_success(f"Removed tag ( {tag}) from term (id: {term_id})")

    @staticmethod
    def _render_guidelines(guidelines: list[Guideline]) -> None:
        guideline_items: list[dict[str, Any]] = [
            {
                "ID": guideline.id,
                "Condition": guideline.condition,
                "Action": guideline.action,
                "Enabled": guideline.enabled,
                "Tags": ", ".join(guideline.tags),
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
                "Role": "Source" if conn.source.id == guideline.id else "Target",
                "Peer Role": "Target" if conn.source.id == guideline.id else "Source",
                "Peer ID": peer.id,
                "Peer Condition": peer.condition,
                "Peer Action": peer.action,
            }

        def to_indirect_entailment_item(conn: GuidelineConnection) -> dict[str, str]:
            return {
                "Connection ID": conn.id,
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
        condition: str,
        action: str,
        tags: list[str],
    ) -> None:
        try:
            guideline = Actions.create_guideline(
                ctx,
                condition,
                action,
                tags=tags,
            )

            Interface._write_success(f"Added guideline (id: {guideline.id})")
            Interface._render_guidelines([guideline])

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def update_guideline(
        ctx: click.Context,
        guideline_id: str,
        condition: str,
        action: str,
    ) -> None:
        try:
            guideline_with_connections = Actions.update_guideline(
                ctx,
                guideline_id,
                condition=condition,
                action=action,
            )

            guideline = guideline_with_connections.guideline
            Interface._write_success(f"Updated guideline (id: {guideline.id})")
            Interface._render_guideline_entailments(
                guideline_with_connections.guideline,
                guideline_with_connections.connections,
                guideline_with_connections.tool_associations,
                include_indirect=False,
            )

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_guideline(
        ctx: click.Context,
        guideline_id: str,
    ) -> None:
        try:
            Actions.delete_guideline(ctx, guideline_id)

            Interface._write_success(f"Removed guideline (id: {guideline_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_guideline(
        ctx: click.Context,
        guideline_id: str,
    ) -> None:
        try:
            guideline_with_connections_and_associations = Actions.view_guideline(ctx, guideline_id)

            Interface._render_guidelines([guideline_with_connections_and_associations.guideline])
            Interface._render_guideline_entailments(
                guideline_with_connections_and_associations.guideline,
                guideline_with_connections_and_associations.connections,
                guideline_with_connections_and_associations.tool_associations,
                include_indirect=True,
            )

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def list_guidelines(
        ctx: click.Context,
        tag: Optional[str],
        hide_disabled: bool,
    ) -> None:
        try:
            guidelines = Actions.list_guidelines(ctx, tag)

            guidelines_to_render = sorted(
                [g for g in guidelines if g.enabled or not hide_disabled],
                key=lambda g: g.enabled or False,
                reverse=True,
            )

            if not guidelines_to_render:
                rich.print(Text("No data available", style="bold yellow"))
                return

            Interface._render_guidelines(guidelines_to_render)

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def create_entailment(
        ctx: click.Context,
        source_guideline_id: str,
        target_guideline_id: str,
    ) -> None:
        try:
            connection = Actions.create_entailment(
                ctx,
                source_guideline_id,
                target_guideline_id,
            )

            Interface._write_success(f"Added connection (id: {connection.connections[0].id})")
            Interface._print_table([connection.dict()])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_entailment(
        ctx: click.Context,
        source_guideline_id: str,
        target_guideline_id: str,
    ) -> None:
        try:
            connection_id = Actions.remove_entailment(
                ctx,
                source_guideline_id,
                target_guideline_id,
            )

            Interface._write_success(f"Removed entailment (id: {connection_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
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
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        try:
            guideline = Actions.add_guideline_tool_association(
                ctx, guideline_id, service_name, tool_name
            )

            Interface._write_success(
                f"Enabled tool '{tool_name}' from service '{service_name}' for guideline '{guideline_id}'"
            )
            Interface._render_guideline_tool_associations(guideline.tool_associations)

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_guideline_tool_association(
        ctx: click.Context,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        try:
            association_id = Actions.remove_guideline_tool_association(
                ctx, guideline_id, service_name, tool_name
            )

            Interface._write_success(f"Removed tool association (id: {association_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def enable_guideline(
        ctx: click.Context,
        guideline_ids: tuple[str],
    ) -> None:
        try:
            guidelines = Actions.enable_guideline(ctx, guideline_ids)

            Interface._write_success(f"Enabled guidelines (ids: {', '.join(guideline_ids)})")

            Interface._render_guidelines(guidelines)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def disable_guideline(
        ctx: click.Context,
        guideline_ids: tuple[str],
    ) -> None:
        try:
            guidelines = Actions.disable_guideline(ctx, guideline_ids)

            Interface._write_success(f"Disabled guidelines (ids: {', '.join(guideline_ids)})")

            Interface._render_guidelines(guidelines)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_guideline_tag(
        ctx: click.Context,
        guideline_id: str,
        tag: str,
    ) -> None:
        try:
            Actions.add_guideline_tag(ctx, guideline_id, tag)
            Interface._write_success(f"Added tag ( {tag}) to guideline (id: {guideline_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_guideline_tag(
        ctx: click.Context,
        guideline_id: str,
        tag: str,
    ) -> None:
        try:
            Actions.remove_guideline_tag(ctx, guideline_id, tag)
            Interface._write_success(f"Removed tag ( {tag}) from guideline (id: {guideline_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_variables(variables: list[ContextVariable]) -> None:
        variable_items = [
            {
                "ID": variable.id,
                "Name": variable.name,
                "Description": variable.description or "",
                "Service Name": variable.tool_id.service_name if variable.tool_id else "",
                "Tool Name": variable.tool_id.tool_name if variable.tool_id else "",
                "Freshness Rules": variable.freshness_rules,
                "Tags": ", ".join(variable.tags or []),
            }
            for variable in variables
        ]

        Interface._print_table(variable_items)

    @staticmethod
    def list_variables(
        ctx: click.Context,
        tag: Optional[str],
    ) -> None:
        variables = Actions.list_variables(ctx, tag)

        if not variables:
            rich.print("No variables found")
            return

        Interface._render_variables(variables)

    @staticmethod
    def create_variable(
        ctx: click.Context,
        name: str,
        description: str,
        service_name: Optional[str],
        tool_name: Optional[str],
        freshness_rules: Optional[str],
        tags: list[str],
    ) -> None:
        variable = Actions.create_variable(
            ctx,
            name,
            description,
            service_name,
            tool_name,
            freshness_rules,
            tags=tags,
        )

        Interface._write_success(f"Added variable (id: {variable.id})")
        Interface._render_variables([variable])

    @staticmethod
    def update_variable(
        ctx: click.Context,
        variable_id: str,
        name: Optional[str],
        description: Optional[str],
        service_name: Optional[str],
        tool_name: Optional[str],
        freshness_rules: Optional[str],
    ) -> None:
        variable = Actions.update_variable(
            ctx, variable_id, name, description, service_name, tool_name, freshness_rules
        )

        Interface._write_success(f"Updated variable (id: {variable.id})")
        Interface._render_variables([variable])

    @staticmethod
    def delete_variable(ctx: click.Context, variable_id: str) -> None:
        try:
            Actions.delete_variable(ctx, variable_id)
            Interface._write_success(f"Removed variable (id: {variable_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
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
        variable_id: str,
        key: str,
        value: str,
    ) -> None:
        try:
            cv_value = Actions.set_variable_value(
                ctx=ctx,
                variable_id=variable_id,
                key=key,
                value=value,
            )

            Interface._write_success(f"Updated variable value (id: {cv_value.id})")
            Interface._render_variable_key_value_pairs({key: cv_value})
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_variable(
        ctx: click.Context,
        variable_id: str,
    ) -> None:
        try:
            read_variable_result = Actions.view_variable(
                ctx,
                variable_id,
                include_values=True,
            )

            Interface._render_variables([read_variable_result.context_variable])

            if not read_variable_result.key_value_pairs:
                rich.print("No values are available")
                return

            pairs: dict[str, ContextVariableValue] = {}
            for k, v in read_variable_result.key_value_pairs.items():
                if v:
                    pairs[k] = v

            Interface._render_variable_key_value_pairs(pairs)

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_variable_value(
        ctx: click.Context,
        variable_id: str,
        key: str,
    ) -> None:
        try:
            value = Actions.view_variable_value(ctx, variable_id, key)

            Interface._render_variable_key_value_pairs({key: value})
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_variable_value(
        ctx: click.Context,
        variable_id: str,
        key: str,
    ) -> None:
        try:
            Actions.delete_variable_value(ctx, variable_id, key)
            Interface._write_success(f"Removed key from variable (id: {variable_id}, key: '{key}')")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_variable_tag(ctx: click.Context, variable_id: str, tag: str) -> None:
        try:
            Actions.add_variable_tag(ctx, variable_id, tag)
            Interface._write_success(f"Added tag ({tag}) to variable (id: {variable_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_variable_tag(ctx: click.Context, variable_id: str, tag: str) -> None:
        try:
            Actions.remove_variable_tag(ctx, variable_id, tag)
            Interface._write_success(f"Removed tag ({tag}) from variable (id: {variable_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def create_service(
        ctx: click.Context,
        name: str,
        kind: str,
        url: str,
        source: str,
        update: bool,
    ) -> None:
        try:
            existing_services = Actions.list_services(ctx)

            if (
                not update
                and next((s for s in existing_services if s.name == name), None) is not None
            ):
                Interface.write_error(f"Error: Service '{name}' already exists")
                set_exit_status(1)
                return

            result = Actions.create_or_update_service(ctx, name, kind, url, source)

            Interface._write_success(f"Added service (name: '{name}')")
            Interface._print_table([result.dict()])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_service(
        ctx: click.Context,
        name: str,
    ) -> None:
        try:
            Actions.delete_service(ctx, name)

            Interface._write_success(f"Removed service (name: '{name}')")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
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
                    rich.print(Text("  Name:", style="bold"), tool.name)
                    if tool.description:
                        rich.print(
                            Text("  Description:\n     ", style="bold"),
                            tool.description,
                        )

                    rich.print(Text("  Parameters:", style="bold"))

                    if tool.parameters:
                        for param_name, param_desc in tool.parameters.items():
                            rich.print(Text(f"    - {param_name}:", style="bold"), end=" ")
                            rich.print(param_desc)
                    else:
                        rich.print("    None")

                    rich.print()
            else:
                rich.print("\nNo tools available for this service.")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_customers(customers: list[Customer]) -> None:
        customer_items: list[dict[str, Any]] = [
            {
                "ID": customer.id,
                "Name": customer.name,
                "Extra": customer.extra,
                "Tags": ", ".join(customer.tags),
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

            Interface._render_customers(customers)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def create_customer(
        ctx: click.Context,
        name: str,
        tags: list[str],
    ) -> None:
        try:
            customer = Actions.create_customer(
                ctx,
                name,
                tags,
            )
            Interface._write_success(f"Added customer (id: {customer.id})")
            Interface._render_customers([customer])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def update_customer(ctx: click.Context, customer_id: str, name: str) -> None:
        try:
            customer = Actions.update_customer(ctx, customer_id=customer_id, name=name)
            Interface._write_success(f"Updated customer (id: {customer_id})")

            Interface._render_customers([customer])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_customer(ctx: click.Context, customer_id: str) -> None:
        try:
            Actions.delete_customer(ctx, customer_id=customer_id)
            Interface._write_success(f"Removed customer (id: {customer_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_customer(ctx: click.Context, customer_id: str) -> None:
        try:
            customer = Actions.view_customer(ctx, customer_id)
            Interface._render_customers([customer])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_customer_extra(ctx: click.Context, customer_id: str, key: str, value: str) -> None:
        try:
            Actions.add_customer_extra(ctx, customer_id, key, value)
            Interface._write_success(
                f"Added extra value to customer (id: {customer_id}, key: '{key}', value: '{value}')"
            )
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_customer_extra(ctx: click.Context, customer_id: str, key: str) -> None:
        try:
            Actions.remove_customer_extra(ctx, customer_id, key)
            Interface._write_success(
                f"Removed extra value from customer (id: {customer_id}, key: '{key}')"
            )
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_customer_tag(
        ctx: click.Context,
        customer_id: str,
        tag: str,
    ) -> None:
        try:
            Actions.add_customer_tag(ctx, customer_id, tag)
            Interface._write_success(f"Tagged customer (id: {customer_id}, tag: {tag})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_customer_tag(
        ctx: click.Context,
        customer_id: str,
        tag: str,
    ) -> None:
        try:
            Actions.remove_customer_tag(ctx, customer_id, tag)
            Interface._write_success(f"Untagged customer (id: {customer_id}, tag: {tag})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_tags(tags: list[Tag]) -> None:
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

            Interface._render_tags(tags)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def create_tag(ctx: click.Context, name: str) -> None:
        try:
            tag = Actions.create_tag(ctx, name=name)
            Interface._write_success(f"Added tag ( {tag.id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_tag(ctx: click.Context, tag: str) -> None:
        try:
            tag_dto = Actions.view_tag(ctx, tag)
            Interface._render_tags([tag_dto])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def update_tag(ctx: click.Context, tag: str, name: str) -> None:
        try:
            tag_dto = Actions.update_tag(ctx, tag=tag, name=name)
            Interface._write_success(f"Updated tag ({tag_dto.id})")

            Interface._render_tags([tag_dto])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_tag(ctx: click.Context, tag: str) -> None:
        try:
            Actions.delete_tag(ctx, tag)
            Interface._write_success(f"Removed tag ({tag})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_utterances(utterances: list[Utterance]) -> None:
        utterance_items = [
            {
                "ID": f.id,
                "Value": f.value,
                "Fields": [
                    f"name: {s.name}, description: {s.description}, examples: {s.examples}"
                    for s in f.fields
                ]
                or "",
                "Tags": ", ".join(f.tags),
                "Creation Date": reformat_datetime(f.creation_utc),
            }
            for f in utterances
        ]

        Interface._print_table(utterance_items)

    @staticmethod
    def load_utterances(ctx: click.Context, path: Path) -> None:
        try:
            utterances = Actions.load_utterances(ctx, path)

            Interface._write_success(f"Loaded {len(utterances)} utterances from {path}")
            Interface._render_utterances(utterances)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def list_utterances(ctx: click.Context) -> None:
        try:
            utterances = Actions.list_utterances(ctx)
            if not utterances:
                rich.print("No utterances found")
                return

            Interface._render_utterances(utterances)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_utterance(ctx: click.Context, utterance_id: str) -> None:
        try:
            utterance = Actions.view_utterance(ctx, utterance_id=utterance_id)
            Interface._render_utterances([utterance])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def stream_logs(
        ctx: click.Context,
        union_patterns: list[str],
        intersection_patterns: list[str],
    ) -> None:
        try:
            for log in Actions.stream_logs(ctx, union_patterns, intersection_patterns):
                level = log.get("level", "")
                message = log.get("message", "")
                correlation_id = log.get("correlation_id", "")
                rich.print(f"[{level}] [{correlation_id}] {message}")
        except Exception as e:
            Interface.write_error(f"Error while streaming logs: {e}")


async def async_main() -> None:
    click_completion.init()  # type: ignore

    @dataclass(frozen=True)
    class Config:
        server_address: str
        client: ParlantClient
        log_server_address: str

    @click.group()
    @click.option(
        "-s",
        "--server",
        type=str,
        help="Server address",
        metavar="ADDRESS[:PORT]",
        default="http://localhost:8800",
    )
    @click.option(
        "--log-port",
        type=int,
        help="Port for the log server",
        metavar="LOG_PORT",
        default=8799,
    )
    @click.pass_context
    def cli(ctx: click.Context, server: str, log_port: int) -> None:
        if not ctx.obj:
            server_url = urlparse(server)
            server_host = server_url.hostname or "localhost"

            log_server_address = f"tcp://{server_host}:{log_port}"

            ctx.obj = Config(
                server_address=server,
                client=ParlantClient(base_url=server),
                log_server_address=log_server_address,
            )

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
    @click.option(
        "--composition-mode",
        type=click.Choice(["fluid", "strict-utterance", "composited-utterance", "fluid-utterance"]),
        help="Composition mode",
        required=False,
    )
    @click.option(
        "--tag",
        type=str,
        metavar="TAG NAME OR ID",
        help="Tag name or ID. May be specified multiple times.",
        required=False,
        multiple=True,
    )
    @click.pass_context
    def agent_create(
        ctx: click.Context,
        name: str,
        description: Optional[str],
        max_engine_iterations: Optional[int],
        composition_mode: Optional[str],
        tag: tuple[str],
    ) -> None:
        if composition_mode:
            composition_mode = composition_mode.replace("-", "_")

        Interface.create_agent(
            ctx=ctx,
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
            composition_mode=composition_mode,
            tags=list(tag),
        )

    @agent.command("delete", help="Delete an agent")
    @click.option("--id", type=str, metavar="ID", help="Agent ID", required=True)
    @click.pass_context
    def agent_remove(ctx: click.Context, id: str) -> None:
        Interface.delete_agent(ctx, id)

    @agent.command("view", help="View an agent")
    @click.option("--id", type=str, metavar="ID", help="Agent ID", required=True)
    @click.pass_context
    def agent_view(ctx: click.Context, id: str) -> None:
        Interface.view_agent(ctx, id)

    @agent.command("list", help="List agents")
    @click.pass_context
    def agent_list(ctx: click.Context) -> None:
        Interface.list_agents(ctx)

    @agent.command("update", help="Update an agent's details")
    @click.option(
        "--id",
        type=str,
        help="Agent ID",
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
    @click.option(
        "--composition-mode",
        "-c",
        type=click.Choice(["fluid", "strict-utterance", "composited-utterance", "fluid-utterance"]),
        help="Composition mode",
        required=False,
    )
    @click.pass_context
    def agent_update(
        ctx: click.Context,
        id: str,
        name: Optional[str],
        description: Optional[str],
        max_engine_iterations: Optional[int],
        composition_mode: Optional[str],
    ) -> None:
        id = id if id else Interface.get_default_agent(ctx)
        assert id

        if composition_mode:
            composition_mode = composition_mode.replace("-", "_")

        Interface.update_agent(ctx, id, name, description, max_engine_iterations, composition_mode)

    @agent.command("tag", help="Tag an agent")
    @click.option("--id", type=str, metavar="ID", help="Agent ID", required=True)
    @click.option("--tag", type=str, metavar="TAG NAME OR ID", help="Tag name or ID", required=True)
    @click.pass_context
    def agent_tag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.add_tag(ctx, id, tag)

    @agent.command("untag", help="Untag an agent")
    @click.option("--id", type=str, metavar="ID", help="Agent ID", required=True)
    @click.option("--tag", type=str, metavar="TAG NAME OR ID", help="Tag name or ID", required=True)
    @click.pass_context
    def agent_remove_tag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.remove_tag(ctx, id, tag)

    @cli.group(help="Manage sessions")
    def session() -> None:
        pass

    @session.command("create", help="Create a session")
    @click.option(
        "--agent-id",
        type=str,
        help="Agent ID",
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
    @click.option("--id", type=str, metavar="ID", help="Session ID", required=True)
    @click.pass_context
    def session_view(ctx: click.Context, id: str) -> None:
        Interface.view_session(ctx, id)

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
    @click.option("--name", type=str, help="Term name", required=True)
    @click.option("--description", type=str, help="Term description", required=True)
    @click.option(
        "--synonyms",
        type=str,
        help="Comma-separated list of synonyms",
        metavar="LIST",
        required=False,
    )
    @click.option(
        "--tag",
        type=str,
        metavar="TAG_NAME_OR_ID",
        help="Tag name or ID. May be specified multiple times.",
        required=False,
        multiple=True,
    )
    @click.pass_context
    def glossary_create(
        ctx: click.Context,
        name: str,
        description: str,
        synonyms: Optional[str],
        tag: tuple[str],
    ) -> None:
        Interface.create_term(
            ctx,
            name,
            description,
            (synonyms or "").split(","),
            list(tag),
        )

    @glossary.command("update", help="Update a term")
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
        id: str,
        name: Optional[str],
        description: Optional[str],
        synonyms: Optional[str],
    ) -> None:
        Interface.update_term(
            ctx,
            id,
            name,
            description,
            (synonyms or "").split(","),
        )

    @glossary.command("delete", help="Delete a term")
    @click.option("--id", type=str, metavar="ID", help="Term ID", required=True)
    @click.pass_context
    def glossary_delete(
        ctx: click.Context,
        id: str,
    ) -> None:
        Interface.delete_term(ctx, id)

    @glossary.command("list", help="List terms")
    @click.option(
        "--tag",
        type=str,
        metavar="TAG_NAME_OR_ID",
        help="Tag name or ID",
        required=False,
    )
    @click.pass_context
    def glossary_list(
        ctx: click.Context,
        tag: Optional[str],
    ) -> None:
        Interface.list_terms(ctx, tag)

    @glossary.command("tag", help="Tag a term")
    @click.option("--id", type=str, metavar="ID", help="Term ID", required=True)
    @click.option("--tag", type=str, metavar="TAG_NAME_OR_ID", help="Tag name or ID", required=True)
    @click.pass_context
    def glossary_tag(
        ctx: click.Context,
        id: str,
        tag: str,
    ) -> None:
        Interface.add_term_tag(
            ctx=ctx,
            term_id=id,
            tag=tag,
        )

    @glossary.command("untag", help="Untag from a term")
    @click.option("--id", type=str, metavar="ID", help="Term ID", required=True)
    @click.option("--tag", type=str, metavar="TAG_NAME_OR_ID", help="Tag name or ID", required=True)
    @click.pass_context
    def glossary_untag(
        ctx: click.Context,
        id: str,
        tag: str,
    ) -> None:
        Interface.remove_term_tag(
            ctx=ctx,
            term_id=id,
            tag=tag,
        )

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
        "--tag",
        type=str,
        metavar="TAG_NAME_OR_ID",
        help="Tag name or ID. May be specified multiple times.",
        required=False,
        multiple=True,
    )
    @click.pass_context
    def guideline_create(
        ctx: click.Context,
        condition: str,
        action: str,
        tag: tuple[str],
    ) -> None:
        Interface.create_guideline(
            ctx=ctx,
            condition=condition,
            action=action,
            tags=list(tag),
        )

    @guideline.command("update", help="Update a guideline")
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.option(
        "--condition",
        type=str,
        help="A statement describing when the guideline should apply",
        required=False,
    )
    @click.option(
        "--action",
        type=str,
        help="The instruction to perform when the guideline applies",
        required=False,
    )
    @click.pass_context
    def guideline_update(
        ctx: click.Context,
        id: str,
        condition: str,
        action: str,
    ) -> None:
        if not (condition or action):
            Interface.write_error("At least one of --condition or --action must be specified")
            raise FastExit()

        Interface.update_guideline(
            ctx=ctx,
            guideline_id=id,
            condition=condition,
            action=action,
        )

    @guideline.command("delete", help="Delete a guideline")
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.pass_context
    def guideline_delete(
        ctx: click.Context,
        id: str,
    ) -> None:
        Interface.delete_guideline(ctx, id)

    @guideline.command("view", help="View a guideline")
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.pass_context
    def guideline_view(
        ctx: click.Context,
        id: str,
    ) -> None:
        Interface.view_guideline(ctx, id)

    @guideline.command("list", help="List guidelines")
    @click.option(
        "--tag",
        type=str,
        metavar="TAG_NAME_OR_ID",
        help="Tag name or ID",
        required=False,
    )
    @click.option(
        "--hide-disabled",
        type=bool,
        show_default=True,
        default=False,
        help="Hide disabled guidelines",
    )
    @click.pass_context
    def guideline_list(
        ctx: click.Context,
        tag: Optional[str],
        hide_disabled: bool,
    ) -> None:
        Interface.list_guidelines(ctx, tag, hide_disabled)

    @guideline.command("entail", help="Create an entailment between two guidelines")
    @click.option("--source", type=str, metavar="ID", help="Source guideline ID", required=True)
    @click.option("--target", type=str, metavar="ID", help="Target guideline ID", required=True)
    @click.pass_context
    def guideline_entail(
        ctx: click.Context,
        source: str,
        target: str,
    ) -> None:
        Interface.create_entailment(
            ctx=ctx,
            source_guideline_id=source,
            target_guideline_id=target,
        )

    @guideline.command("disentail", help="Delete an entailment between two guidelines")
    @click.option("--source", type=str, metavar="ID", help="Source guideline ID", required=True)
    @click.option("--target", type=str, metavar="ID", help="Target guideline ID", required=True)
    @click.pass_context
    def guideline_disentail(
        ctx: click.Context,
        source: str,
        target: str,
    ) -> None:
        Interface.remove_entailment(
            ctx=ctx,
            source_guideline_id=source,
            target_guideline_id=target,
        )

    @guideline.command("tool-enable", help="Allow a guideline to make use of a tool")
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.option(
        "--service",
        type=str,
        metavar="NAME",
        help="The name of the tool service containing the tool",
        required=True,
    )
    @click.option("--tool", type=str, metavar="NAME", help="Tool name", required=True)
    @click.pass_context
    def guideline_enable_tool(
        ctx: click.Context,
        id: str,
        service: str,
        tool: str,
    ) -> None:
        Interface.add_guideline_tool_association(
            ctx=ctx,
            guideline_id=id,
            service_name=service,
            tool_name=tool,
        )

    @guideline.command("tool-disable", help="Disallow a guideline to make use of a tool")
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.option(
        "--service",
        type=str,
        metavar="NAME",
        help="The name of the tool service containing the tool",
        required=True,
    )
    @click.option("--tool", type=str, metavar="NAME", help="Tool name", required=True)
    @click.pass_context
    def guideline_disable_tool(
        ctx: click.Context,
        id: str,
        service: str,
        tool: str,
    ) -> None:
        Interface.remove_guideline_tool_association(
            ctx=ctx,
            guideline_id=id,
            service_name=service,
            tool_name=tool,
        )

    @guideline.command("enable", help="Enable a guideline")
    @click.option(
        "--id",
        "ids",
        type=str,
        metavar="ID",
        help="Guideline ID, May be specified multiple times.",
        required=True,
        multiple=True,
    )
    @click.pass_context
    def guideline_enable(
        ctx: click.Context,
        ids: tuple[str],
    ) -> None:
        Interface.enable_guideline(
            ctx=ctx,
            guideline_ids=ids,
        )

    @guideline.command("disable", help="Disable a guideline")
    @click.option(
        "--id",
        "ids",
        type=str,
        metavar="ID",
        help="Guideline ID, May be specified multiple times.",
        required=True,
        multiple=True,
    )
    @click.pass_context
    def guideline_disable(
        ctx: click.Context,
        ids: tuple[str],
    ) -> None:
        Interface.disable_guideline(
            ctx=ctx,
            guideline_ids=ids,
        )

    @guideline.command("tag", help="Tag a guideline")
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.option("--tag", type=str, metavar="TAG_NAME_OR_ID", help="Tag name or ID", required=True)
    @click.pass_context
    def guideline_tag(
        ctx: click.Context,
        id: str,
        tag: str,
    ) -> None:
        Interface.add_guideline_tag(
            ctx=ctx,
            guideline_id=id,
            tag=tag,
        )

    @guideline.command("untag", help="Untag from a guideline")
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.option("--tag", type=str, metavar="TAG_NAME_OR_ID", help="Tag name or ID", required=True)
    @click.pass_context
    def guideline_untag(
        ctx: click.Context,
        id: str,
        tag: str,
    ) -> None:
        Interface.remove_guideline_tag(
            ctx=ctx,
            guideline_id=id,
            tag=tag,
        )

    @cli.group(help="Manage an agent's context variables")
    def variable() -> None:
        pass

    @variable.command("list", help="List variables")
    @click.option(
        "--tag",
        type=str,
        metavar="TAG_NAME_OR_ID",
        help="Tag name or ID",
        required=False,
    )
    @click.pass_context
    def variable_list(
        ctx: click.Context,
        tag: Optional[str],
    ) -> None:
        Interface.list_variables(
            ctx=ctx,
            tag=tag,
        )

    @variable.command("create", help="Create a context variable")
    @click.option("--description", type=str, help="Variable description", required=False)
    @click.option("--name", type=str, metavar="NAME", help="Variable name", required=True)
    @click.option(
        "--service",
        type=str,
        metavar="NAME",
        help="The name of the tool service containing the tool",
        required=False,
    )
    @click.option("--tool", type=str, metavar="NAME", help="Tool name", required=False)
    @click.option("--freshness-rules", type=str, help="Variable freshness rules", required=False)
    @click.option(
        "--tag",
        type=str,
        metavar="TAG_NAME_OR_ID",
        help="Tag name or ID. May be specified multiple times.",
        required=False,
        multiple=True,
    )
    @click.pass_context
    def variable_create(
        ctx: click.Context,
        name: str,
        description: Optional[str],
        service: Optional[str],
        tool: Optional[str],
        freshness_rules: Optional[str],
        tag: tuple[str],
    ) -> None:
        if service or tool:
            assert service
            assert tool

        Interface.create_variable(
            ctx=ctx,
            name=name,
            description=description or "",
            service_name=service,
            tool_name=tool,
            freshness_rules=freshness_rules,
            tags=list(tag),
        )

    @variable.command("update", help="Update a context variable")
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @click.option("--description", type=str, help="Variable description", required=False)
    @click.option("--name", type=str, metavar="NAME", help="Variable name", required=False)
    @click.option(
        "--service",
        type=str,
        metavar="NAME",
        help="The name of the tool service containing the tool",
        required=False,
    )
    @click.option("--tool", type=str, metavar="NAME", help="Tool name", required=False)
    @click.option("--freshness-rules", type=str, help="Variable freshness rules", required=False)
    @click.pass_context
    def variable_update(
        ctx: click.Context,
        id: str,
        name: Optional[str],
        description: Optional[str],
        service: Optional[str],
        tool: Optional[str],
        freshness_rules: Optional[str],
    ) -> None:
        if service or tool:
            assert service
            assert tool

        Interface.update_variable(
            ctx=ctx,
            variable_id=id,
            name=name,
            description=description or "",
            service_name=service,
            tool_name=tool,
            freshness_rules=freshness_rules,
        )

    @variable.command("delete", help="Delete a context variable")
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @click.pass_context
    def variable_delete(
        ctx: click.Context,
        id: str,
    ) -> None:
        Interface.delete_variable(
            ctx=ctx,
            variable_id=id,
        )

    @variable.command("set", help="Set the value of a key under a context variable")
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @click.option(
        "--key",
        type=str,
        metavar="NAME",
        help='The key (e.g. <CUSTOMER_ID> or "tag:<TAG_ID>" or "DEFAULT" to set a default value)',
    )
    @click.option("--value", type=str, metavar="TEXT", help="The key's value")
    @click.pass_context
    def variable_set(
        ctx: click.Context,
        id: str,
        key: str,
        value: str,
    ) -> None:
        Interface.set_variable_value(
            ctx=ctx,
            variable_id=id,
            key=key,
            value=value,
        )

    @variable.command("get", help="Get the value(s) of a variable")
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @click.option(
        "--key",
        type=str,
        metavar="NAME",
        help='The key (e.g. <CUSTOMER_ID> or "tag:<TAG_ID>" or "DEFAULT" to set a default value)',
    )
    @click.pass_context
    def variable_get(
        ctx: click.Context,
        id: str,
        key: Optional[str],
    ) -> None:
        if key:
            Interface.view_variable_value(
                ctx=ctx,
                variable_id=id,
                key=key,
            )
        else:
            Interface.view_variable(
                ctx=ctx,
                variable_id=id,
            )

    @variable.command("delete-value", help="Delete a context variable value")
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @click.option(
        "--key",
        type=str,
        metavar="NAME",
        help='The key (e.g. <CUSTOMER_ID> or "tag:<TAG_ID>" or "DEFAULT" to set a default value)',
    )
    @click.pass_context
    def variable_value_delete(
        ctx: click.Context,
        id: str,
        key: str,
    ) -> None:
        Interface.delete_variable_value(
            ctx=ctx,
            variable_id=id,
            key=key,
        )

    @variable.command("tag", help="Tag a variable")
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @click.option("--tag", type=str, metavar="TAG_NAME_OR_ID", help="Tag name or ID", required=True)
    @click.pass_context
    def variable_tag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.add_variable_tag(ctx, id, tag)

    @variable.command("untag", help="Untag a variable")
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @click.option("--tag", type=str, metavar="TAG_NAME_OR_ID", help="Tag name or ID", required=True)
    @click.pass_context
    def variable_untag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.remove_variable_tag(ctx, id, tag)

    @cli.group(help="Manage services")
    def service() -> None:
        pass

    @service.command("create", help="Create a service")
    @click.option(
        "--kind",
        type=click.Choice(["sdk", "openapi"]),
        required=True,
        help="Service kind",
    )
    @click.option(
        "--url",
        metavar="URL",
        required=True,
        help="Service URL",
    )
    @click.option(
        "--source",
        required=False,
        metavar="SOURCE",
        help="For an OpenAPI service, this is the local path or URL to its openapi.json",
    )
    @click.option("--name", type=str, metavar="NAME", help="Service name", required=True)
    @click.pass_context
    def service_create(
        ctx: click.Context,
        name: str,
        kind: str,
        url: str,
        source: str,
    ) -> None:
        Interface.create_service(ctx, name, kind, url, source, False)

    @service.command("update", help="Update a service")
    @click.option(
        "--kind",
        type=click.Choice(["sdk", "openapi"]),
        required=True,
        help="Service kind",
    )
    @click.option(
        "--url",
        metavar="URL",
        required=True,
        help="Service URL",
    )
    @click.option(
        "--source",
        required=False,
        metavar="SOURCE",
        help="For an OpenAPI service, this is the local path or URL to its openapi.json",
    )
    @click.option("--name", type=str, metavar="NAME", help="Service name", required=True)
    @click.pass_context
    def service_update(
        ctx: click.Context,
        name: str,
        kind: str,
        url: str,
        source: str,
    ) -> None:
        Interface.create_service(ctx, name, kind, url, source, True)

    @service.command("delete", help="Delete a service")
    @click.option("--name", type=str, metavar="NAME", help="Service name", required=True)
    @click.pass_context
    def service_delete(ctx: click.Context, name: str) -> None:
        Interface.delete_service(ctx, name)

    @service.command("list", help="List services")
    @click.pass_context
    def service_list(ctx: click.Context) -> None:
        Interface.list_services(ctx)

    @service.command("view", help="View a service and its tools")
    @click.option("--name", type=str, metavar="NAME", help="Service name", required=True)
    @click.pass_context
    def service_view(ctx: click.Context, name: str) -> None:
        Interface.view_service(ctx, name)

    @cli.group(help="Manage customers")
    def customer() -> None:
        pass

    @customer.command("create", help="Create a customer")
    @click.option("--name", type=str, metavar="NAME", help="Customer name", required=True)
    @click.option(
        "--tag",
        type=str,
        metavar="TAG NAME OR ID",
        help="Tag name or ID. May be specified multiple times.",
        required=False,
        multiple=True,
    )
    @click.pass_context
    def customer_create(
        ctx: click.Context,
        name: str,
        tag: tuple[str],
    ) -> None:
        Interface.create_customer(
            ctx,
            name,
            list(tag),
        )

    @customer.command("list", help="List customers")
    @click.pass_context
    def customer_list(ctx: click.Context) -> None:
        Interface.list_customers(ctx)

    @customer.command("update", help="Update a customer")
    @click.option("--id", type=str, metavar="ID", help="Customer ID", required=True)
    @click.option("--name", type=str, metavar="NAME", help="Customer name", required=True)
    @click.pass_context
    def customer_update(ctx: click.Context, id: str, name: str) -> None:
        Interface.update_customer(ctx, id, name)

    @customer.command("delete", help="Delete a customer")
    @click.option("--id", type=str, metavar="ID", help="Customer ID", required=True)
    @click.pass_context
    def customer_delete(ctx: click.Context, id: str) -> None:
        Interface.delete_customer(ctx, id)

    @customer.command("view", help="View a customer")
    @click.option("--id", type=str, metavar="ID", help="Customer ID", required=True)
    @click.pass_context
    def customer_view(ctx: click.Context, id: str) -> None:
        Interface.view_customer(ctx, id)

    @customer.command("set", help="Set extra info for a customer using a key and value")
    @click.option("--id", type=str, metavar="ID", help="Customer ID", required=True)
    @click.option(
        "--key",
        type=str,
        metavar="NAME",
        help="The key of the property (e.g. 'email')",
        required=True,
    )
    @click.option("--value", type=str, metavar="TEXT", help="The key's value")
    @click.pass_context
    def customer_set(ctx: click.Context, id: str, key: str, value: str) -> None:
        Interface.add_customer_extra(ctx, id, key, value)

    @customer.command("unset", help="Unset extra info for a customer")
    @click.option("--id", type=str, metavar="ID", help="Customer ID", required=True)
    @click.option(
        "--key",
        type=str,
        metavar="NAME",
        help="The key of the property (e.g. 'email')",
        required=True,
    )
    @click.pass_context
    def customer_unset(ctx: click.Context, id: str, key: str) -> None:
        Interface.remove_customer_extra(ctx, id, key)

    @customer.command("tag", help="Tag a customer")
    @click.option("--id", type=str, metavar="ID", help="Customer ID", required=True)
    @click.option("--tag", type=str, metavar="TAG_NAME_OR_ID", help="Tag name or ID", required=True)
    @click.pass_context
    def customer_tag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.add_customer_tag(ctx, id, tag)

    @customer.command("untag", help="Untag a customer")
    @click.option("--id", type=str, metavar="ID", help="Customer ID", required=True)
    @click.option("--tag", type=str, metavar="TAG_NAME_OR_ID", help="Tag name or ID", required=True)
    @click.pass_context
    def customer_untag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.remove_customer_tag(ctx, id, tag)

    @cli.group(help="Manage tags")
    def tag() -> None:
        """Group of commands to manage tags."""

    @tag.command("list", help="List tags")
    @click.pass_context
    def tag_list(ctx: click.Context) -> None:
        Interface.list_tags(ctx)

    @tag.command("create", help="Create a tag")
    @click.option("--name", type=str, metavar="NAME", help="Tag name", required=True)
    @click.pass_context
    def tag_create(ctx: click.Context, name: str) -> None:
        Interface.create_tag(ctx, name)

    @tag.command("view", help="View a tag")
    @click.option("--id", type=str, metavar="ID", help="Tag ID", required=True)
    @click.pass_context
    def tag_view(ctx: click.Context, id: str) -> None:
        Interface.view_tag(ctx, id)

    @tag.command("update", help="Update a tag")
    @click.option("--id", type=str, metavar="ID", help="Tag ID", required=True)
    @click.option("--name", type=str, metavar="NAME", help="Tag name", required=True)
    @click.pass_context
    def tag_update(ctx: click.Context, id: str, name: str) -> None:
        Interface.update_tag(ctx, id, name)

    @tag.command("delete", help="Delete a tag")
    @click.option("--id", type=str, metavar="ID", help="Tag ID", required=True)
    @click.pass_context
    def tag_delete(ctx: click.Context, id: str) -> None:
        Interface.delete_tag(ctx, id)

    @cli.group(help="Manage utterances")
    def utterance() -> None:
        pass

    @utterance.command("init", help="Initialize a sample utterances JSON file.")
    @click.argument("file", type=click.Path(dir_okay=False, writable=True))
    def utterance_init(file: str) -> None:
        sample_data = {
            "utterances": [
                {
                    "value": "Hello, {{username}}!",
                    "tags": [],
                },
                {
                    "value": "Your balance is {{balance}}",
                },
            ]
        }

        path = Path(file).resolve()
        if path.exists():
            rich.print(Text(f"Overwriting existing file at {path}", style="bold yellow"))

        with path.open("w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2)

        Interface._write_success(f"Created sample utterance data at {path}")

    @utterance.command("load", help="Load utterances from a JSON file.")
    @click.argument("file", type=click.Path(exists=True, dir_okay=False))
    @click.pass_context
    def utterance_load(ctx: click.Context, file: str) -> None:
        Interface.load_utterances(ctx, Path(file))

    @utterance.command("list", help="List utterances")
    @click.pass_context
    def utterance_list(ctx: click.Context) -> None:
        Interface.list_utterances(ctx)

    @utterance.command("view", help="View an utterance")
    @click.option("--id", type=str, metavar="ID", help="Utterance ID", required=True)
    @click.pass_context
    def utterance_view(ctx: click.Context, id: str) -> None:
        Interface.view_utterance(ctx, id)

    @cli.command(
        "log",
        help="Stream server logs",
    )
    @click.option(
        "--guideline-matcher", "-g", is_flag=True, help="Filter logs by [GuidelineMatcher]"
    )
    @click.option("--tool-caller", "-t", is_flag=True, help="Filter logs by [ToolCaller]")
    @click.option(
        "--message-event-composer",
        "-m",
        is_flag=True,
        help="Filter logs by [MessageEventComposer]",
    )
    @click.option(
        "-a",
        "--and",
        "intersection_patterns",
        multiple=True,
        default=[],
        metavar="PATTERN",
        help="Patterns to intersect with. May be specified multiple times.",
    )
    @click.option(
        "-o",
        "--or",
        "union_patterns",
        multiple=True,
        default=[],
        metavar="PATTERN",
        help="Patterns to union by. May be specified multiple times.",
    )
    @click.pass_context
    def log_view(
        ctx: click.Context,
        guideline_matcher: bool,
        tool_caller: bool,
        message_event_composer: bool,
        intersection_patterns: tuple[str],
        union_patterns: tuple[str],
    ) -> None:
        union_pattern_list = list(union_patterns)

        if guideline_matcher:
            union_pattern_list.append("[GuidelineMatcher]")
        if tool_caller:
            union_pattern_list.append("[ToolCaller]")
        if message_event_composer:
            union_pattern_list.append("[MessageEventComposer]")

        Interface.stream_logs(ctx, union_pattern_list, list(intersection_patterns))

    @cli.command(
        "help",
        context_settings={"ignore_unknown_options": True},
        help="Show help for a command",
    )
    @click.argument("command", nargs=-1, required=False)
    @click.pass_context
    def help_command(ctx: click.Context, command: Optional[tuple[str]] = None) -> None:
        def transform_and_exec_help(command: str) -> None:
            new_args = [sys.argv[0]] + command.split() + ["--help"]
            os.execvp(sys.executable, [sys.executable] + new_args)

        if not command:
            click.echo(cli.get_help(ctx))
        else:
            transform_and_exec_help(" ".join(command))

    cli()


def main() -> None:
    async def wrapped_main() -> None:
        try:
            await async_main()
        except ApiError as e:
            try:
                Interface.write_error(f"Error: {e.body['detail']}")
            except KeyError:
                Interface.write_error(f"Error: Uncaught API error: status-code={e.status_code}")
            set_exit_status(1)
        except FastExit:
            pass

    asyncio.run(wrapped_main())


if __name__ == "__main__":
    main()
    sys.exit(get_exit_status())
