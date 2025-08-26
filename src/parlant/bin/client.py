# Copyright 2025 Emcie Co Ltd.
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
from contextlib import suppress
import json
import os
from pathlib import Path
import time
from urllib.parse import urlparse
import click
from dataclasses import dataclass
from datetime import datetime
import requests
import rich
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TaskProgressColumn
from rich import box
from rich.table import Table
from rich.text import Text
import sys
from typing import Any, Callable, Iterator, Optional, OrderedDict, cast

from parlant.client import ParlantClient
from parlant.client.core import ApiError
from parlant.client.types import (
    Agent,
    AgentTagUpdateParams,
    Capability,
    CapabilityTagUpdateParams,
    CannedResponse,
    CannedResponseField,
    ConsumptionOffsetsUpdateParams,
    ContextVariable,
    ContextVariableReadResult,
    ContextVariableValue,
    ContextVariableTagsUpdateParams,
    Customer,
    CustomerMetadataUpdateParams,
    CustomerTagUpdateParams,
    Event,
    Journey,
    JourneyTagUpdateParams,
    JourneyConditionUpdateParams,
    Guideline,
    Relationship,
    RelationshipKindDto,
    GuidelinePayload,
    GuidelineContent,
    GuidelineToolAssociation,
    GuidelineToolAssociationUpdateParams,
    GuidelineTagsUpdateParams,
    GuidelineWithRelationshipsAndToolAssociations,
    GuidelineMetadataUpdateParams,
    OpenApiServiceParams,
    Payload,
    SdkServiceParams,
    McpServiceParams,
    Service,
    Session,
    Term,
    TermTagsUpdateParams,
    Tool,
    ToolId,
    Tag,
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

        if tag.startswith("agent:"):
            agent_id = tag.split(":")[1]
            if client.agents.retrieve(agent_id):
                return tag
            else:
                raise Exception(f"Agent (id: {agent_id}) not found")

        if tag.startswith("journey:"):
            journey_id = tag.split(":")[1]
            if client.journeys.retrieve(journey_id):
                return tag
            else:
                raise Exception(f"Journey (id: {journey_id}) not found")

        tags = client.tags.list()
        for t in tags:
            if t.name == tag or t.id == tag:
                return t.id

        raise Exception(f"Tag ({tag}) not found")

    @staticmethod
    def _fetch_tool_id(
        ctx: click.Context,
        tool_id: ToolId,
    ) -> ToolId:
        client = cast(ParlantClient, ctx.obj.client)
        try:
            service = client.services.retrieve(tool_id.service_name)
        except Exception:
            raise Exception(f"Service ({tool_id.service_name}) not found")

        if next((t for t in service.tools or [] if t.name == tool_id.tool_name), None):
            return tool_id

        raise Exception(f"Tool ({tool_id.tool_name}) not found in service ({tool_id.service_name})")

    @staticmethod
    def _parse_relationship_side(
        ctx: click.Context,
        entity_id: str,
    ) -> tuple[str | ToolId, str]:
        with suppress(Exception):
            if tag_id := Actions._fetch_tag_id(ctx, entity_id):
                return tag_id, "tag"

        with suppress(Exception):
            if ":" in entity_id and (
                tool_id := Actions._fetch_tool_id(
                    ctx,
                    ToolId(service_name=entity_id.split(":")[0], tool_name=entity_id.split(":")[1]),
                )
            ):
                return tool_id, "tool"

        client = cast(ParlantClient, ctx.obj.client)
        client.guidelines.retrieve(entity_id)
        return entity_id, "guideline"

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
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.agents.update(
            agent_id=agent_id,
            tags=AgentTagUpdateParams(add=[tag_id]),
        )

        return tag_id

    @staticmethod
    def remove_tag(
        ctx: click.Context,
        agent_id: str,
        tag: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.agents.update(
            agent_id=agent_id,
            tags=AgentTagUpdateParams(remove=[tag_id]),
        )

        return tag_id

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
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.glossary.update_term(term_id, tags=TermTagsUpdateParams(add=[tag_id]))

        return tag_id

    @staticmethod
    def remove_term_tag(
        ctx: click.Context,
        term_id: str,
        tag: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.glossary.update_term(term_id, tags=TermTagsUpdateParams(remove=[tag_id]))

        return tag_id

    @staticmethod
    def create_guideline(
        ctx: click.Context,
        condition: str,
        action: Optional[str],
        tool_id: Optional[str],
        tags: list[str],
    ) -> GuidelineWithRelationshipsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)
        tags = list(set([Actions._fetch_tag_id(ctx, t) for t in tags]))

        tool_ids = (
            [
                Actions._fetch_tool_id(
                    ctx, ToolId(service_name=tool_id.split(":")[0], tool_name=tool_id.split(":")[1])
                )
            ]
            if tool_id
            else []
        )

        evaluation = client.evaluations.create(
            payloads=[
                Payload(
                    kind="guideline",
                    guideline=GuidelinePayload(
                        content=GuidelineContent(condition=condition),
                        tool_ids=tool_ids,
                        operation="add",
                        action_proposition=True,
                        properties_proposition=True,
                    ),
                )
            ]
        )

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(style="bold blue"),
            "{task.completed}/{task.total}",
            TimeElapsedColumn(),
        ) as progress:
            progress_task = progress.add_task("Evaluating guideline\n", total=100)

            while True:
                time.sleep(0.2)
                evaluation_result = client.evaluations.retrieve(
                    evaluation.id,
                    wait_for_completion=0,
                )

                if evaluation_result.status in ["pending", "running"]:
                    progress.update(progress_task, completed=int(evaluation_result.progress))
                    continue

                if evaluation_result.status == "completed":
                    progress.update(progress_task, completed=100)

                    invoice = evaluation_result.invoices[0]
                    assert invoice.approved
                    assert invoice.data
                    assert invoice.data.guideline
                    assert invoice.payload.guideline

                    guideline = client.guidelines.create(
                        condition=condition,
                        action=action if action else invoice.data.guideline.action_proposition,
                        tags=tags,
                        metadata=invoice.data.guideline.properties_proposition or {},
                    )

                    guideline_with_relationships_and_associations = client.guidelines.update(
                        guideline.id,
                        tool_associations=GuidelineToolAssociationUpdateParams(
                            add=tool_ids,
                        ),
                    )

                    return guideline_with_relationships_and_associations

                elif evaluation_result.status == "failed":
                    raise ValueError(evaluation_result.error)

        if tool_id:
            tool_id_obj = Actions._fetch_tool_id(
                ctx, ToolId(service_name=tool_id.split(":")[0], tool_name=tool_id.split(":")[1])
            )

            guideline_with_relationships_and_associations = client.guidelines.update(
                guideline_id=guideline.id,
                tool_associations=GuidelineToolAssociationUpdateParams(
                    add=[tool_id_obj],
                ),
            )

            return guideline_with_relationships_and_associations

        return GuidelineWithRelationshipsAndToolAssociations(
            guideline=guideline,
            relationships=[],
            tool_associations=[],
        )

    @staticmethod
    def update_guideline(
        ctx: click.Context,
        guideline_id: str,
        condition: Optional[str] = None,
        action: Optional[str] = None,
    ) -> GuidelineWithRelationshipsAndToolAssociations:
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
    ) -> GuidelineWithRelationshipsAndToolAssociations:
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
    def add_guideline_tool_association(
        ctx: click.Context,
        guideline_id: str,
        service_name: str,
        tool_name: str,
    ) -> GuidelineWithRelationshipsAndToolAssociations:
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
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.guidelines.update(guideline_id, tags=GuidelineTagsUpdateParams(add=[tag_id]))

        return tag_id

    @staticmethod
    def remove_guideline_tag(
        ctx: click.Context,
        guideline_id: str,
        tag: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.guidelines.update(guideline_id, tags=GuidelineTagsUpdateParams(remove=[tag_id]))

        return tag_id

    @staticmethod
    def set_guideline_metadata(
        ctx: click.Context,
        guideline_id: str,
        key: str,
        value: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.guidelines.update(
            guideline_id,
            metadata=GuidelineMetadataUpdateParams(add={key: value}),
        )

    @staticmethod
    def unset_guideline_metadata(
        ctx: click.Context,
        guideline_id: str,
        key: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.guidelines.update(
            guideline_id,
            metadata=GuidelineMetadataUpdateParams(remove=[key]),
        )

    @staticmethod
    def create_relationship(
        ctx: click.Context,
        source: str,
        target: str,
        kind: RelationshipKindDto,
    ) -> Relationship:
        client = cast(ParlantClient, ctx.obj.client)

        source_id, source_type = Actions._parse_relationship_side(ctx, source)
        target_id, target_type = Actions._parse_relationship_side(ctx, target)

        return client.relationships.create(
            source_guideline=cast(str, source_id) if source_type == "guideline" else None,
            source_tag=cast(str, source_id) if source_type == "tag" else None,
            source_tool=cast(ToolId, source_id) if source_type == "tool" else None,
            target_guideline=cast(str, target_id) if target_type == "guideline" else None,
            target_tag=cast(str, target_id) if target_type == "tag" else None,
            target_tool=cast(ToolId, target_id) if target_type == "tool" else None,
            kind=kind,
        )

    @staticmethod
    def remove_relationship(
        ctx: click.Context,
        id: Optional[str],
        source_id: Optional[str],
        target_id: Optional[str],
        kind: Optional[RelationshipKindDto],
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        if id:
            client.relationships.delete(id)
            return id

        assert source_id and target_id and kind

        _, source_type = Actions._parse_relationship_side(ctx, source_id)

        if relationship := next(
            (
                r
                for r in client.relationships.list(
                    guideline_id=source_id if source_type == "guideline" else None,
                    tag_id=source_id if source_type == "tag" else None,
                    tool_id=source_id if source_type == "tool" else None,
                    kind=kind,
                    indirect=False,
                )
                if (
                    (r.source_guideline and source_id == r.source_guideline.id)
                    or (r.source_tag and source_id == r.source_tag.id)
                    or (r.source_tool and source_id.split(":")[1] == r.source_tool.name)
                )
                and (
                    (r.target_guideline and target_id == r.target_guideline.id)
                    or (r.target_tag and target_id == r.target_tag.id)
                    or (r.target_tool and target_id.split(":")[1] == r.target_tool.name)
                )
                and r.kind == kind
            ),
            None,
        ):
            client.relationships.delete(relationship.id)

            return relationship.id

        raise ValueError(
            f"A relationship between {source_id} and {target_id} with kind {kind} was not found"
        )

    @staticmethod
    def list_relationships(
        ctx: click.Context,
        guideline_id: Optional[str],
        tag: Optional[str],
        tool_id: Optional[str],
        kind: Optional[RelationshipKindDto],
        indirect: Optional[bool],
    ) -> list[Relationship]:
        client = cast(ParlantClient, ctx.obj.client)

        tag_id = Actions._fetch_tag_id(ctx, tag) if tag else None
        if tool_id:
            Actions._fetch_tool_id(
                ctx, ToolId(service_name=tool_id.split(":")[0], tool_name=tool_id.split(":")[1])
            )

        return client.relationships.list(
            guideline_id=guideline_id,
            tag_id=tag_id,
            tool_id=tool_id,
            kind=kind,
            indirect=indirect,
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

        if key.startswith("tag:"):
            tag_spec = key.split(":")[1]
            tag_id = Actions._fetch_tag_id(ctx, tag_spec)
            key = f"tag:{tag_id}"

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

        if key.startswith("tag:"):
            tag_spec = key.split(":")[1]
            tag_id = Actions._fetch_tag_id(ctx, tag_spec)
            key = f"tag:{tag_id}"

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
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.context_variables.update(
            variable_id, tags=ContextVariableTagsUpdateParams(add=[tag_id])
        )
        return tag_id

    @staticmethod
    def remove_variable_tag(
        ctx: click.Context,
        variable_id: str,
        tag: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.context_variables.update(
            variable_id,
            tags=ContextVariableTagsUpdateParams(remove=[tag_id]),
        )
        return tag_id

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

        elif kind == "mcp":
            result = client.services.create_or_update(
                name=name,
                kind="mcp",
                mcp=McpServiceParams(url=url),
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
            metadata={},
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
    def add_customer_metadata(
        ctx: click.Context,
        customer_id: str,
        key: str,
        value: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.customers.update(
            customer_id=customer_id, metadata=CustomerMetadataUpdateParams(set={key: value})
        )

    @staticmethod
    def remove_customer_metadata(
        ctx: click.Context,
        customer_id: str,
        key: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.customers.update(
            customer_id=customer_id, metadata=CustomerMetadataUpdateParams(unset=[key])
        )

    @staticmethod
    def add_customer_tag(
        ctx: click.Context,
        customer_id: str,
        tag: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.customers.update(
            customer_id=customer_id,
            tags=CustomerTagUpdateParams(add=[tag_id]),
        )

        return tag_id

    @staticmethod
    def remove_customer_tag(
        ctx: click.Context,
        customer_id: str,
        tag: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.customers.update(
            customer_id=customer_id,
            tags=CustomerTagUpdateParams(remove=[tag_id]),
        )

        return tag_id

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
        tag_id = Actions._fetch_tag_id(ctx, tag)

        client = cast(ParlantClient, ctx.obj.client)
        return client.tags.retrieve(tag_id=tag_id)

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
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.tags.delete(tag_id=tag_id)

        return tag_id

    @staticmethod
    def view_tool(
        ctx: click.Context,
        tool_id: str,
    ) -> Tool:
        client = cast(ParlantClient, ctx.obj.client)

        tool_id_obj = Actions._fetch_tool_id(
            ctx,
            ToolId(service_name=tool_id.split(":")[0], tool_name=tool_id.split(":")[1]),
        )

        service = client.services.retrieve(tool_id_obj.service_name)

        if tool := next((t for t in service.tools or [] if t.name == tool_id_obj.tool_name), None):
            return tool
        else:
            raise Exception(
                f"Tool ({tool_id_obj.tool_name}) not found in service ({tool_id_obj.service_name})"
            )

    @staticmethod
    def list_canned_responses(ctx: click.Context) -> list[CannedResponse]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.canned_responses.list()

    @staticmethod
    def view_canned_response(ctx: click.Context, canned_response_id: str) -> CannedResponse:
        client = cast(ParlantClient, ctx.obj.client)
        return client.canned_responses.retrieve(canned_response_id=canned_response_id)

    @staticmethod
    def load_canned_responses(ctx: click.Context, path: Path) -> list[CannedResponse]:
        with open(path, "r") as file:
            data = json.load(file)

        client = cast(ParlantClient, ctx.obj.client)

        for canned_response in client.canned_responses.list():
            client.canned_responses.delete(canned_response_id=canned_response.id)

        canned_responses = []
        tag_ids = {tag.name: tag.id for tag in client.tags.list()}

        for canned_response_data in data.get("canned_responses", []):
            value = canned_response_data["value"]
            assert value

            fields = [
                CannedResponseField(**canned_response_field)
                for canned_response_field in canned_response_data.get("fields", [])
            ]

            tag_names = canned_response_data.get("tags", [])

            signals = canned_response_data.get("signals", [])

            canned_response = client.canned_responses.create(
                value=value,
                fields=fields,
                tags=[tag_ids[tag_name] for tag_name in tag_names if tag_name in tag_ids] or None,
                signals=signals,
            )

            canned_responses.append(canned_response)

        return canned_responses

    @staticmethod
    def list_journeys(
        ctx: click.Context,
        tag: Optional[str] = None,
    ) -> list[Journey]:
        client = cast(ParlantClient, ctx.obj.client)
        if tag:
            return client.journeys.list(tag_id=Actions._fetch_tag_id(ctx, tag))
        else:
            return client.journeys.list()

    @staticmethod
    def create_journey(
        ctx: click.Context,
        title: str,
        description: str,
        conditions: list[str],
        tags: list[str],
    ) -> Journey:
        client = cast(ParlantClient, ctx.obj.client)

        journey = client.journeys.create(
            title=title,
            description=description,
            conditions=conditions,
            tags=tags,
        )

        return journey

    @staticmethod
    def view_journey(
        ctx: click.Context,
        journey_id: str,
    ) -> Journey:
        client = cast(ParlantClient, ctx.obj.client)
        return client.journeys.retrieve(journey_id=journey_id)

    @staticmethod
    def update_journey(
        ctx: click.Context,
        journey_id: str,
        title: str,
        description: str,
    ) -> Journey:
        client = cast(ParlantClient, ctx.obj.client)

        return client.journeys.update(
            journey_id=journey_id,
            title=title,
            description=description,
        )

    @staticmethod
    def delete_journey(
        ctx: click.Context,
        journey_id: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.journeys.delete(journey_id=journey_id)

    @staticmethod
    def add_journey_condition(
        ctx: click.Context,
        journey_id: str,
        guideline_id: Optional[str],
        condition: Optional[str],
    ) -> Journey:
        client = cast(ParlantClient, ctx.obj.client)

        guideline_id = (
            guideline_id
            or client.guidelines.create(
                condition=cast(str, condition),
                metadata={"journeys": [journey_id]},
            ).id
        )

        return client.journeys.update(
            journey_id=journey_id,
            conditions=JourneyConditionUpdateParams(add=[guideline_id]),
        )

    @staticmethod
    def remove_journey_condition(
        ctx: click.Context,
        journey_id: str,
        guideline_id: str,
    ) -> Journey:
        client = cast(ParlantClient, ctx.obj.client)

        return client.journeys.update(
            journey_id=journey_id,
            conditions=JourneyConditionUpdateParams(remove=[guideline_id]),
        )

    @staticmethod
    def add_journey_tag(
        ctx: click.Context,
        journey_id: str,
        tag: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.journeys.update(journey_id=journey_id, tags=JourneyTagUpdateParams(add=[tag_id]))

        return tag_id

    @staticmethod
    def remove_journey_tag(
        ctx: click.Context,
        journey_id: str,
        tag: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.journeys.update(journey_id=journey_id, tags=JourneyTagUpdateParams(remove=[tag_id]))

        return tag_id

    @staticmethod
    def create_capability(
        ctx: click.Context,
        title: str,
        description: str,
        signals: list[str],
        tags: list[str],
    ) -> Capability:
        client = cast(ParlantClient, ctx.obj.client)
        tags = list(set([Actions._fetch_tag_id(ctx, t) for t in tags]))

        return client.capabilities.create(
            title=title,
            description=description,
            signals=signals,
            tags=tags,
        )

    @staticmethod
    def update_capability(
        ctx: click.Context,
        capability_id: str,
        title: Optional[str],
        description: Optional[str],
        signals: Optional[list[str]],
    ) -> Capability:
        client = cast(ParlantClient, ctx.obj.client)

        return client.capabilities.update(
            capability_id=capability_id,
            title=title,
            description=description,
            signals=signals,
        )

    @staticmethod
    def view_capability(
        ctx: click.Context,
        capability_id: str,
    ) -> Capability:
        client = cast(ParlantClient, ctx.obj.client)

        return client.capabilities.retrieve(
            capability_id=capability_id,
        )

    @staticmethod
    def list_capabilities(
        ctx: click.Context,
        tag: Optional[str],
    ) -> list[Capability]:
        client = cast(ParlantClient, ctx.obj.client)

        if tag:
            return client.capabilities.list(tag_id=Actions._fetch_tag_id(ctx, tag))
        else:
            return client.capabilities.list()

    @staticmethod
    def delete_capability(
        ctx: click.Context,
        capability_id: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)

        client.capabilities.delete(capability_id=capability_id)

    @staticmethod
    def add_capability_tag(
        ctx: click.Context,
        capability_id: str,
        tag: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.capabilities.update(capability_id, tags=CapabilityTagUpdateParams(add=[tag_id]))

        return tag_id

    @staticmethod
    def remove_capability_tag(
        ctx: click.Context,
        capability_id: str,
        tag: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        tag_id = Actions._fetch_tag_id(ctx, tag)
        client.capabilities.update(
            capability_id,
            tags=CapabilityTagUpdateParams(remove=[tag_id]),
        )

        return tag_id

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
    def _print_table(
        data: list[dict[str, Any]],
        header_order: list[str] | None = None,
    ) -> None:
        """Render a list of dictionaries as a rich table.

        If *header_order* is provided, the columns will appear in that order (filtered
        down to only the headers present in *data*). Any headers found in *data* that
        are **not** present in *header_order* will be appended to the end in the order
        of their first appearance.  This allows callers to enforce a consistent column
        ordering while still gracefully handling extra/unknown fields.
        """

        table = Table(box=box.ROUNDED, border_style="bright_green")

        table.add_column("#", header_style="bright_green", overflow="fold")

        if not data:
            rich.print(table)
            return

        # Collect all headers that actually appear in *data* (their first appearance
        # defines the fallback ordering for any that are not covered by *header_order*).
        discovered_headers: list[str] = list(
            OrderedDict({key: None for entry in data for key in entry.keys()}).keys()
        )

        if header_order is not None:
            # Keep only headers that exist in the data
            ordered_headers = [h for h in header_order if h in discovered_headers]

            # Append any remaining headers discovered in data that were not specified
            # by *header_order* – preserving discovery order so as not to surprise.
            ordered_headers.extend([h for h in discovered_headers if h not in ordered_headers])
        else:
            ordered_headers = discovered_headers

        for header in ordered_headers:
            table.add_column(header, header_style="bright_green", overflow="fold")

        for idx, row in enumerate(data, start=1):
            row_values = [str(row.get(h, "")) for h in ordered_headers]
            table.add_row(str(idx), *row_values)

        rich.print(table)

    @staticmethod
    def _render_agents(agents: list[Agent]) -> None:
        agent_items: list[dict[str, Any]] = [
            {
                "ID": a.id,
                "Name": a.name,
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
            tag_id = Actions.add_tag(ctx, agent_id, tag)
            Interface._write_success(f"Tagged agent (id: {agent_id}, tag_id: {tag_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_tag(ctx: click.Context, agent_id: str, tag: str) -> None:
        try:
            tag_id = Actions.remove_tag(ctx, agent_id, tag)
            Interface._write_success(f"Untagged agent (id: {agent_id}, tag_id: {tag_id})")
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
        tag_id = Actions.add_term_tag(ctx, term_id, tag)
        Interface._write_success(f"Added tag (id: {tag_id}) to term (id: {term_id})")

    @staticmethod
    def remove_term_tag(
        ctx: click.Context,
        term_id: str,
        tag: str,
    ) -> None:
        tag_id = Actions.remove_term_tag(ctx, term_id, tag)
        Interface._write_success(f"Removed tag (id: {tag_id}) from term (id: {term_id})")

    @staticmethod
    def _render_guidelines(guidelines: list[Guideline]) -> None:
        guideline_items: list[dict[str, Any]] = [
            {
                "ID": guideline.id,
                "Condition": guideline.condition,
                "Action": (
                    guideline.action
                    if guideline.action
                    else f"Activate journey(s): {', '.join(tag.split('journey:')[1] for tag in guideline.tags if tag.startswith('journey:'))}"
                    if any(tag for tag in guideline.tags if tag.startswith("journey:"))
                    else "None"
                ),
                "Enabled": guideline.enabled,
                "Tags": ", ".join(guideline.tags),
                "Metadata": ", ".join([f"{k}: {v}" for k, v in guideline.metadata.items()])
                if guideline.metadata
                else "",
            }
            for guideline in guidelines
        ]

        Interface._print_table(guideline_items)

    @staticmethod
    def _render_relationships(
        entity: Guideline | Tag | Tool | None,
        relationships: list[Relationship],
        include_indirect: bool,
    ) -> None:
        def to_direct_relationship_item(rel: Relationship) -> dict[str, str]:
            result: dict[str, str] = {
                "Relationship ID": rel.id,
                "Kind": rel.kind,
            }

            if rel.source_guideline:
                result.update(
                    {
                        "Source ID": rel.source_guideline.id,
                        "Source Type": "Guideline",
                        "Source Condition": rel.source_guideline.condition,
                        "Source Action": rel.source_guideline.action or "",
                    }
                )
            elif rel.source_tag:
                assert rel.source_tag is not None
                result.update(
                    {
                        "Source ID": rel.source_tag.id,
                        "Source Type": "Tag",
                        "Source Name": rel.source_tag.name,
                    }
                )
            elif rel.source_tool:
                assert rel.source_tool is not None
                result.update(
                    {
                        "Source Type": "Tool",
                        "Source Name": rel.source_tool.name,
                    }
                )
            if rel.target_guideline:
                result.update(
                    {
                        "Target ID": rel.target_guideline.id,
                        "Target Type": "Guideline",
                        "Target Condition": rel.target_guideline.condition,
                        "Target Action": rel.target_guideline.action or "",
                    }
                )
            elif rel.target_tag:
                assert rel.target_tag is not None
                result.update(
                    {
                        "Target ID": rel.target_tag.id,
                        "Target Type": "Tag",
                        "Target Name": rel.target_tag.name,
                    }
                )
            elif rel.target_tool:
                assert rel.target_tool is not None
                result.update(
                    {
                        "Target Type": "Tool",
                        "Target Name": rel.target_tool.name,
                    }
                )

            return result

        def to_indirect_relationship_item(rel: Relationship) -> dict[str, str]:
            result: dict[str, str] = {
                "Relationship ID": rel.id,
                "Kind": rel.kind,
            }

            if rel.source_guideline:
                result.update(
                    {
                        "Source ID": rel.source_guideline.id,
                        "Source Type": "Guideline",
                        "Source Condition": rel.source_guideline.condition,
                        "Source Action": rel.source_guideline.action or "",
                    }
                )
            elif rel.source_tag:
                result.update(
                    {
                        "Source ID": rel.source_tag.id,
                        "Source Type": "Tag",
                        "Source Name": rel.source_tag.name,
                    }
                )
            elif rel.source_tool:
                result.update(
                    {
                        "Source Type": "Tool",
                        "Source Name": rel.source_tool.name,
                    }
                )
            if rel.target_guideline:
                result.update(
                    {
                        "Target ID": rel.target_guideline.id,
                        "Target Type": "Guideline",
                        "Target Condition": rel.target_guideline.condition,
                        "Target Action": rel.target_guideline.action or "",
                    }
                )
            elif rel.target_tag:
                result.update(
                    {
                        "Target ID": rel.target_tag.id,
                        "Target Type": "Tag",
                        "Target Name": rel.target_tag.name,
                    }
                )
            elif rel.target_tool:
                result.update(
                    {
                        "Target Type": "Tool",
                        "Target Name": rel.target_tool.name,
                    }
                )
            return result

        if relationships:
            direct = [
                r
                for r in relationships
                if entity
                in (
                    r.source_guideline,
                    r.target_guideline,
                    r.source_tag,
                    r.target_tag,
                    r.source_tool,
                    r.target_tool,
                )
            ]

            indirect = [r for r in relationships if r not in direct]

            if direct:
                rich.print("Direct Relationships:")

                # Pre-calculate dictionary view of the relationships.
                direct_items = list(map(lambda r: to_direct_relationship_item(r), direct))

                # Determine a consistent column ordering for the *direct* view so
                # that headers like "Source Name" appear next to other "Source"-
                # prefixed fields irrespective of which relationship type happens
                # to be listed first.
                all_direct_keys = {key for entry in direct_items for key in entry.keys()}

                base_order = ["Relationship ID", "Kind"]
                src_tgt_suffixes = ["ID", "Type", "Name", "Condition", "Action"]

                preferred_order: list[str] = base_order.copy()
                for prefix in ("Source", "Target"):
                    for suffix in src_tgt_suffixes:
                        header = f"{prefix} {suffix}"
                        if header in all_direct_keys:
                            preferred_order.append(header)

                Interface._print_table(
                    direct_items,
                    header_order=preferred_order,
                )

            if indirect and include_indirect:
                rich.print("\nIndirect Relationships:")

                indirect_items = list(map(lambda r: to_indirect_relationship_item(r), indirect))

                all_indirect_keys = {key for entry in indirect_items for key in entry.keys()}

                base_order = ["Relationship ID", "Kind"]
                source_target_suffixes = ["ID", "Type", "Name", "Condition", "Action"]
                preferred_order_indirect: list[str] = base_order.copy()
                for prefix in ("Source", "Target"):
                    for suffix in source_target_suffixes:
                        header = f"{prefix} {suffix}"
                        if header in all_indirect_keys:
                            preferred_order_indirect.append(header)

                Interface._print_table(
                    indirect_items,
                    header_order=preferred_order_indirect,
                )

    @staticmethod
    def create_guideline(
        ctx: click.Context,
        condition: str,
        action: Optional[str],
        tool_id: Optional[str],
        tags: tuple[str],
    ) -> None:
        try:
            guideline_with_relationships_and_associations = Actions.create_guideline(
                ctx,
                condition,
                action,
                tool_id,
                tags=list(tags),
            )

            Interface._write_success(
                f"Added guideline (id: {guideline_with_relationships_and_associations.guideline.id})"
            )
            Interface._render_guidelines([guideline_with_relationships_and_associations.guideline])
            Interface._render_relationships(
                guideline_with_relationships_and_associations.guideline,
                guideline_with_relationships_and_associations.relationships,
                include_indirect=False,
            )
            Interface._render_guideline_tool_associations(
                guideline_with_relationships_and_associations.tool_associations
            )

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
            guideline_with_relationships_and_associations = Actions.update_guideline(
                ctx,
                guideline_id,
                condition=condition,
                action=action,
            )

            guideline = guideline_with_relationships_and_associations.guideline
            Interface._write_success(f"Updated guideline (id: {guideline.id})")
            Interface._render_relationships(
                guideline_with_relationships_and_associations.guideline,
                guideline_with_relationships_and_associations.relationships,
                include_indirect=False,
            )
            Interface._render_guideline_tool_associations(
                guideline_with_relationships_and_associations.tool_associations
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
            guideline_with_relationships_and_associations = Actions.view_guideline(
                ctx, guideline_id
            )

            Interface._render_guidelines([guideline_with_relationships_and_associations.guideline])
            Interface._render_relationships(
                guideline_with_relationships_and_associations.guideline,
                guideline_with_relationships_and_associations.relationships,
                include_indirect=True,
            )
            Interface._render_guideline_tool_associations(
                guideline_with_relationships_and_associations.tool_associations
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
    def _render_guideline_tool_associations(
        associations: list[GuidelineToolAssociation],
    ) -> None:
        if associations:
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
            tag_id = Actions.add_guideline_tag(ctx, guideline_id, tag)
            Interface._write_success(f"Added tag (id: {tag_id}) to guideline (id: {guideline_id})")
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
            tag_id = Actions.remove_guideline_tag(ctx, guideline_id, tag)
            Interface._write_success(
                f"Removed tag (id: {tag_id}) from guideline (id: {guideline_id})"
            )
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def set_guideline_metadata(
        ctx: click.Context,
        guideline_id: str,
        key: str,
        value: str,
    ) -> None:
        try:
            Actions.set_guideline_metadata(ctx, guideline_id, key, value)
            Interface._write_success(
                f"Added metadata (key: {key}, value: {value}) to guideline (id: {guideline_id})"
            )
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def unset_guideline_metadata(
        ctx: click.Context,
        guideline_id: str,
        key: str,
    ) -> None:
        try:
            Actions.unset_guideline_metadata(ctx, guideline_id, key)
            Interface._write_success(
                f"Removed metadata (key: {key}) from guideline (id: {guideline_id})"
            )
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def create_relationship(
        ctx: click.Context,
        source_id: str,
        target_id: str,
        kind: RelationshipKindDto,
    ) -> None:
        try:
            relationship = Actions.create_relationship(
                ctx,
                source_id,
                target_id,
                kind,
            )

            Interface._write_success(f"Added relationship (id: {relationship.id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_relationship(
        ctx: click.Context,
        id: Optional[str],
        source_id: Optional[str],
        target_id: Optional[str],
        kind: Optional[RelationshipKindDto],
    ) -> None:
        try:
            relationship_id = Actions.remove_relationship(
                ctx,
                id,
                source_id,
                target_id,
                kind,
            )

            Interface._write_success(f"Removed relationship (id: {relationship_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def list_relationships(
        ctx: click.Context,
        guideline_id: Optional[str],
        tag: Optional[str],
        tool_id: Optional[str],
        kind: Optional[RelationshipKindDto],
        indirect: Optional[bool],
    ) -> None:
        try:
            relationships = Actions.list_relationships(
                ctx,
                guideline_id=guideline_id,
                tag=tag,
                tool_id=tool_id,
                kind=kind,
                indirect=indirect,
            )

            if not relationships:
                rich.print(Text("No data available", style="bold yellow"))
                return

            entity: Guideline | Tag | Tool | None = None
            if guideline_id:
                entity = Actions.view_guideline(ctx, guideline_id).guideline
            elif tag:
                entity = Actions.view_tag(ctx, tag)
            elif tool_id:
                entity = Actions.view_tool(ctx, tool_id)

            Interface._render_relationships(
                entity,
                relationships,
                include_indirect=indirect or True,
            )

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
            tag_id = Actions.add_variable_tag(ctx, variable_id, tag)
            Interface._write_success(f"Added tag (id: {tag_id}) to variable (id: {variable_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_variable_tag(ctx: click.Context, variable_id: str, tag: str) -> None:
        try:
            tag_id = Actions.remove_variable_tag(ctx, variable_id, tag)
            Interface._write_success(
                f"Removed tag (id: {tag_id}) from variable (id: {variable_id})"
            )
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
                "Metadata": customer.metadata,
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
            Actions.add_customer_metadata(ctx, customer_id, key, value)
            Interface._write_success(
                f"Added extra value to customer (id: {customer_id}, key: '{key}', value: '{value}')"
            )
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_customer_extra(ctx: click.Context, customer_id: str, key: str) -> None:
        try:
            Actions.remove_customer_metadata(ctx, customer_id, key)
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
            tag_id = Actions.add_customer_tag(ctx, customer_id, tag)
            Interface._write_success(f"Tagged customer (id: {customer_id}, tag_id: {tag_id})")
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
            tag_id = Actions.remove_customer_tag(ctx, customer_id, tag)
            Interface._write_success(f"Untagged customer (id: {customer_id}, tag_id: {tag_id})")
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
            Interface._write_success(f"Added tag (id: {tag.id})")
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
            Interface._write_success(f"Updated tag (id: {tag_dto.id})")

            Interface._render_tags([tag_dto])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_tag(ctx: click.Context, tag: str) -> None:
        try:
            tag_id = Actions.delete_tag(ctx, tag)
            Interface._write_success(f"Removed tag (id: {tag_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_canned_responses(canreps: list[CannedResponse]) -> None:
        canned_response_items = [
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
            for f in canreps
        ]

        Interface._print_table(canned_response_items)

    @staticmethod
    def load_canned_responses(ctx: click.Context, path: Path) -> None:
        try:
            canned_responses = Actions.load_canned_responses(ctx, path)

            Interface._write_success(f"Loaded {len(canned_responses)} canned_responses from {path}")
            Interface._render_canned_responses(canned_responses)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def list_canned_responses(ctx: click.Context) -> None:
        try:
            canreps = Actions.list_canned_responses(ctx)
            if not canreps:
                rich.print("No canned responses found")
                return

            Interface._render_canned_responses(canreps)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_canned_response(ctx: click.Context, canned_response_id: str) -> None:
        try:
            canned_response = Actions.view_canned_response(
                ctx, canned_response_id=canned_response_id
            )
            Interface._render_canned_responses([canned_response])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_journeys(journeys: list[Journey]) -> None:
        journey_items: list[dict[str, Any]] = [
            {
                "ID": journey.id,
                "Title": journey.title,
                "Description": journey.description,
                "Condition Guideline IDs": ", ".join(journey.conditions),
                "Tags": ", ".join(journey.tags or []),
            }
            for journey in journeys
        ]

        Interface._print_table(journey_items)

    @staticmethod
    def list_journeys(
        ctx: click.Context,
        tag: Optional[str],
    ) -> None:
        try:
            journeys = Actions.list_journeys(ctx, tag)

            if not journeys:
                rich.print(Text("No data available", style="bold yellow"))
                return

            Interface._render_journeys(journeys)

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def create_journey(
        ctx: click.Context,
        title: str,
        description: str,
        conditions: list[str],
        tags: list[str],
    ) -> None:
        try:
            journey = Actions.create_journey(ctx, title, description, conditions, tags)
            Interface._write_success(f"Created journey (id: {journey.id})")
            Interface._render_journeys([journey])

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def update_journey(
        ctx: click.Context,
        journey_id: str,
        title: str,
        description: str,
    ) -> None:
        try:
            journey = Actions.update_journey(ctx, journey_id, title, description)
            Interface._write_success(f"Updated journey (id: {journey.id})")
            Interface._render_journeys([journey])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_journey_condition(
        ctx: click.Context,
        journey_id: str,
        guideline_id: Optional[str],
        condition: Optional[str],
    ) -> None:
        try:
            journey = Actions.add_journey_condition(ctx, journey_id, guideline_id, condition)
            Interface._write_success(f"Added condition to journey (id: {journey.id})")
            Interface._render_journeys([journey])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_journey_condition(
        ctx: click.Context,
        journey_id: str,
        guideline_id: str,
    ) -> None:
        try:
            journey = Actions.remove_journey_condition(ctx, journey_id, guideline_id)
            Interface._write_success(f"Removed condition from journey (id: {journey.id})")
            Interface._render_journeys([journey])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_journey_tag(
        ctx: click.Context,
        journey_id: str,
        tag: str,
    ) -> None:
        try:
            tag_id = Actions.add_journey_tag(ctx, journey_id, tag)
            Interface._write_success(f"Added tag (id: {tag_id}) to journey (id: {journey_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_journey_tag(
        ctx: click.Context,
        journey_id: str,
        tag: str,
    ) -> None:
        try:
            tag_id = Actions.remove_journey_tag(ctx, journey_id, tag)
            Interface._write_success(f"Removed tag (id: {tag_id}) from journey (id: {journey_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_journey(ctx: click.Context, journey_id: str) -> None:
        try:
            Actions.delete_journey(ctx, journey_id)
            Interface._write_success(f"Deleted journey (id: {journey_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_capabilities(capabilities: list[Capability]) -> None:
        items = [
            {
                "ID": c.id,
                "Title": c.title,
                "Description": c.description,
                "Signals": ", ".join(c.signals),
                "Tags": ", ".join(c.tags or []),
            }
            for c in capabilities
        ]
        Interface._print_table(items)

    @staticmethod
    def create_capability(
        ctx: click.Context,
        title: str,
        description: str,
        signals: list[str],
        tags: list[str],
    ) -> None:
        try:
            capability = Actions.create_capability(ctx, title, description, signals, tags)

            Interface._write_success(f"Added capability (id: {getattr(capability, 'id', '')})")

            Interface._render_capabilities([capability])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def update_capability(
        ctx: click.Context,
        capability_id: str,
        title: Optional[str],
        description: Optional[str],
        signals: Optional[list[str]],
    ) -> None:
        try:
            capability = Actions.update_capability(ctx, capability_id, title, description, signals)

            Interface._write_success(f"Updated capability (id: {getattr(capability, 'id', '')})")

            Interface._render_capabilities([capability])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_capability(
        ctx: click.Context,
        capability_id: str,
    ) -> None:
        try:
            capability = Actions.view_capability(ctx, capability_id)

            Interface._render_capabilities([capability])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def list_capabilities(ctx: click.Context, tag: Optional[str]) -> None:
        try:
            capabilities = Actions.list_capabilities(ctx, tag)

            if not capabilities:
                rich.print(Text("No data available", style="bold yellow"))
                return

            Interface._render_capabilities(capabilities)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_capability(ctx: click.Context, capability_id: str) -> None:
        try:
            Actions.delete_capability(ctx, capability_id)

            Interface._write_success(f"Removed capability (id: {capability_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_capability_tag(
        ctx: click.Context,
        capability_id: str,
        tag: str,
    ) -> None:
        try:
            tag_id = Actions.add_capability_tag(ctx, capability_id, tag)

            Interface._write_success(
                f"Added tag (id: {tag_id}) to capability (id: {capability_id})"
            )
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_capability_tag(
        ctx: click.Context,
        capability_id: str,
        tag: str,
    ) -> None:
        try:
            tag_id = Actions.remove_capability_tag(ctx, capability_id, tag)

            Interface._write_success(
                f"Removed tag (id: {tag_id}) from capability (id: {capability_id})"
            )
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
            set_exit_status(1)


def tag_option(
    required: bool = False,
    multiple: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        return click.option(
            "--tag",
            type=str,
            metavar="TAG_NAME | TAG_ID",
            help="Tag name or ID. May be specified multiple times.",
            required=required,
            multiple=multiple,
        )(f)

    return decorator


async def async_main() -> None:
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
        type=click.Choice(
            [
                "fluid",
                "strict_canned",
                "composited_canned",
                "canned_fluid",
            ]
        ),
        help="Composition mode",
        required=False,
    )
    @tag_option(multiple=True)
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
        type=click.Choice(
            [
                "fluid",
                "strict_canned",
                "composited_canned",
                "canned_fluid",
            ]
        ),
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
    @tag_option(required=True)
    @click.pass_context
    def agent_tag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.add_tag(ctx, id, tag)

    @agent.command("untag", help="Untag an agent")
    @click.option("--id", type=str, metavar="ID", help="Agent ID", required=True)
    @tag_option(required=True)
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
    @tag_option(required=False, multiple=True)
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
    @tag_option()
    @click.pass_context
    def glossary_list(
        ctx: click.Context,
        tag: Optional[str],
    ) -> None:
        Interface.list_terms(ctx, tag)

    @glossary.command("tag", help="Tag a term")
    @click.option("--id", type=str, metavar="ID", help="Term ID", required=True)
    @tag_option(required=True)
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
    @tag_option(required=True)
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
        required=False,
    )
    @click.option(
        "--tool-id",
        type=str,
        help="The ID of the tool to associate with the guideline, in the format service_name:tool_name",
        required=False,
    )
    @tag_option(multiple=True)
    @click.pass_context
    def guideline_create(
        ctx: click.Context,
        condition: str,
        action: Optional[str],
        tool_id: Optional[str],
        tag: tuple[str],
    ) -> None:
        Interface.create_guideline(
            ctx=ctx,
            condition=condition,
            action=action,
            tool_id=tool_id,
            tags=tag,
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
            set_exit_status(1)
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
    @tag_option()
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

    @guideline.command("tool-enable", help="Allow a guideline to make use of a tool")
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=False)
    @click.option(
        "--service",
        type=str,
        metavar="NAME",
        help="The name of the tool service containing the tool",
        required=True,
    )
    @click.option("--tool", type=str, metavar="NAME", help="Tool name", required=False)
    @click.option(
        "--tool-id",
        type=str,
        metavar="ID",
        help="Tool ID. format: service_name:tool_name",
        required=False,
    )
    @click.pass_context
    def guideline_enable_tool(
        ctx: click.Context,
        id: str,
        service: Optional[str],
        tool: Optional[str],
        tool_id: Optional[str],
    ) -> None:
        if not (service and tool) and not tool_id:
            Interface.write_error(
                "At least one of --service, --tool, or --tool-id must be specified"
            )
            set_exit_status(1)
            raise FastExit()

        if service and tool and tool_id:
            Interface.write_error("Only one of --service, --tool, or --tool-id can be specified")
            set_exit_status(1)
            raise FastExit()

        if tool_id:
            service_name, tool_name = tool_id.split(":")
        else:
            assert service and tool
            service_name = service
            tool_name = tool

        Interface.add_guideline_tool_association(
            ctx=ctx,
            guideline_id=id,
            service_name=service_name,
            tool_name=tool_name,
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
    @tag_option(required=True)
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
    @tag_option(required=True)
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

    @guideline.command("set", help="Set metadata for a guideline using a key and value")
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.option("--key", type=str, metavar="KEY", help="Key", required=True)
    @click.option("--value", type=str, metavar="VALUE", help="Value", required=True)
    @click.pass_context
    def guideline_set(ctx: click.Context, id: str, key: str, value: str) -> None:
        Interface.set_guideline_metadata(ctx, id, key, value)

    @guideline.command("unset", help="Remove metadata for a guideline using a key")
    @click.option("--id", type=str, metavar="ID", help="Guideline ID", required=True)
    @click.option("--key", type=str, metavar="KEY", help="Key", required=True)
    @click.pass_context
    def guideline_unset(ctx: click.Context, id: str, key: str) -> None:
        Interface.unset_guideline_metadata(ctx, id, key)

    @cli.group(help="Manage relationships")
    def relationship() -> None:
        pass

    @relationship.command("create", help="Create a relationship")
    @click.option(
        "--source",
        type=str,
        metavar="TAG_NAME | TAG_ID | GUIDELINE_ID | TOOL_ID",
        help="Source tag or guideline ID or tool ID",
        required=True,
    )
    @click.option(
        "--target",
        type=str,
        metavar="TAG_NAME | TAG_ID | GUIDELINE_ID | TOOL_ID",
        help="Target tag or guideline ID or tool ID",
        required=True,
    )
    @click.option(
        "--kind",
        type=click.Choice(
            [
                "entailment",
                "priority",
                "dependency",
                "disambiguation",
                "reevaluation",
                "overlap",
            ]
        ),
        help="Relationship kind",
        required=True,
    )
    @click.pass_context
    def relationship_create(
        ctx: click.Context,
        source: str,
        target: str,
        kind: RelationshipKindDto,
    ) -> None:
        Interface.create_relationship(
            ctx=ctx,
            source_id=source,
            target_id=target,
            kind=kind,
        )

    @relationship.command("delete", help="Delete a relationship between two guidelines")
    @click.option("--id", type=str, metavar="ID", help="Relationship ID")
    @click.option(
        "--source",
        type=str,
        metavar="GUIDELINE_ID",
        help="Source of the relationship",
    )
    @click.option(
        "--target",
        type=str,
        metavar="TAG_NAME | TAG_ID | GUIDELINE_ID",
        help="Target tag or guideline ID",
    )
    @click.option(
        "--kind",
        type=click.Choice(
            [
                "entailment",
                "priority",
                "dependency",
                "disambiguation",
                "reevaluation",
                "overlap",
            ]
        ),
        help="Relationship kind",
    )
    @click.pass_context
    def relationship_delete(
        ctx: click.Context,
        id: Optional[str],
        source: Optional[str],
        target: Optional[str],
        kind: Optional[RelationshipKindDto],
    ) -> None:
        if id:
            if source or target or kind:
                Interface.write_error("When --id is used, other identifiers must not be used")
                set_exit_status(1)
                raise FastExit()
        if source or target or kind:
            if id:
                Interface.write_error("When specifying source and target, ID must not be specified")
            if not (source and target and kind):
                Interface.write_error("Please specify --source, --target, and --kind")

        Interface.remove_relationship(
            ctx=ctx,
            id=id,
            source_id=source,
            target_id=target,
            kind=kind,
        )

    @relationship.command("list", help="List relationships")
    @click.option(
        "--kind",
        type=click.Choice(
            [
                "entailment",
                "priority",
                "dependency",
                "disambiguation",
                "reevaluation",
                "overlap",
            ]
        ),
        help="Relationship kind",
        required=False,
    )
    @click.option(
        "--guideline-id",
        type=str,
        metavar="GUIDELINE_ID",
        help="Guideline ID",
        required=False,
    )
    @click.option(
        "--tool",
        type=str,
        metavar="TOOL_ID",
        help="Tool ID, format: service_name:tool_name",
    )
    @tag_option(required=False)
    @click.option(
        "--indirect",
        type=bool,
        help="Include indirect relationships. Default is true.",
        required=False,
        default=True,
    )
    @click.pass_context
    def relationship_list(
        ctx: click.Context,
        guideline_id: Optional[str],
        tag: Optional[str],
        tool: Optional[str],
        kind: Optional[RelationshipKindDto],
        indirect: Optional[bool],
    ) -> None:
        if guideline_id and tag:
            Interface.write_error("Either --guideline-id or --tag must be provided, not both")
            set_exit_status(1)
            raise FastExit()

        Interface.list_relationships(ctx, guideline_id, tag, tool, kind, indirect)

    @cli.group(help="Manage an agent's context variables")
    def variable() -> None:
        pass

    @variable.command("list", help="List variables")
    @tag_option()
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
    @tag_option(multiple=True)
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
    @tag_option(required=True)
    @click.pass_context
    def variable_tag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.add_variable_tag(ctx, id, tag)

    @variable.command("untag", help="Untag a variable")
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def variable_untag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.remove_variable_tag(ctx, id, tag)

    @cli.group(help="Manage services")
    def service() -> None:
        pass

    @service.command("create", help="Create a service")
    @click.option(
        "--kind",
        type=click.Choice(["sdk", "openapi", "mcp"]),
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
        type=click.Choice(["sdk", "openapi", "mcp"]),
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
    @tag_option(multiple=True)
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
    @tag_option(required=True)
    @click.pass_context
    def customer_tag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.add_customer_tag(ctx, id, tag)

    @customer.command("untag", help="Untag a customer")
    @click.option("--id", type=str, metavar="ID", help="Customer ID", required=True)
    @tag_option(required=True)
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
    @tag_option(required=True)
    @click.pass_context
    def tag_view(ctx: click.Context, tag: str) -> None:
        Interface.view_tag(ctx, tag)

    @tag.command("update", help="Update a tag")
    @click.option("--id", type=str, metavar="ID", help="Tag ID", required=True)
    @click.option("--name", type=str, metavar="NAME", help="Tag name", required=True)
    @click.pass_context
    def tag_update(ctx: click.Context, id: str, name: str) -> None:
        Interface.update_tag(ctx, id, name)

    @tag.command("delete", help="Delete a tag")
    @tag_option(required=True)
    @click.pass_context
    def tag_delete(ctx: click.Context, tag: str) -> None:
        Interface.delete_tag(ctx, tag)

    @cli.group(help="Manage canned responses")
    def canned_response() -> None:
        pass

    @canned_response.command("init", help="Initialize a sample canned responses JSON file.")
    @click.argument("file", type=click.Path(dir_okay=False, writable=True))
    def canned_response_init(file: str) -> None:
        sample_data = {
            "canned_responses": [
                {
                    "value": "Hello, {{std.customer.name}}!",
                },
                {
                    "value": "My name is {{std.agent.name}}",
                },
            ]
        }

        path = Path(file).resolve()
        if path.exists():
            rich.print(Text(f"Overwriting existing file at {path}", style="bold yellow"))

        with path.open("w", encoding="utf-8") as f:
            json.dump(sample_data, f, indent=2)

        Interface._write_success(f"Created sample canned response data at {path}")

    @canned_response.command("load", help="Load canned responses from a JSON file.")
    @click.argument("file", type=click.Path(exists=True, dir_okay=False))
    @click.pass_context
    def canned_response_load(ctx: click.Context, file: str) -> None:
        Interface.load_canned_responses(ctx, Path(file))

    @canned_response.command("list", help="List canned responses")
    @click.pass_context
    def canned_response_list(ctx: click.Context) -> None:
        Interface.list_canned_responses(ctx)

    @canned_response.command("view", help="View an canned_response")
    @click.option("--id", type=str, metavar="ID", help="Canned Response ID", required=True)
    @click.pass_context
    def canned_response_view(ctx: click.Context, id: str) -> None:
        Interface.view_canned_response(ctx, id)

    @cli.group(help="Manage journeys")
    def journey() -> None:
        pass

    @journey.command("list", help="List journeys")
    @tag_option(multiple=True)
    @click.pass_context
    def journey_list(
        ctx: click.Context,
        tag: Optional[str],
    ) -> None:
        Interface.list_journeys(ctx, tag)

    @journey.command("create", help="Create a journey")
    @click.option("--title", type=str, metavar="TITLE", help="Journey title", required=True)
    @click.option(
        "--description",
        type=str,
        metavar="DESCRIPTION",
        help="Journey description. can be multiple lines",
        required=True,
    )
    @click.option(
        "--condition",
        type=str,
        metavar="CONDITION",
        help="Journey conditions",
        multiple=True,
        required=True,
    )
    @tag_option(multiple=True)
    @click.pass_context
    def journey_create(
        ctx: click.Context,
        title: str,
        description: str,
        condition: tuple[str],
        tag: tuple[str],
    ) -> None:
        Interface.create_journey(
            ctx=ctx,
            title=title,
            description=description,
            conditions=list(condition),
            tags=list(tag),
        )

    @journey.command("update", help="Update a journey")
    @click.option("--id", type=str, metavar="ID", help="Journey ID", required=True)
    @click.option("--title", type=str, metavar="TITLE", help="Journey title", required=True)
    @click.option(
        "--description", type=str, metavar="DESCRIPTION", help="Journey description", required=True
    )
    @click.pass_context
    def journey_update(ctx: click.Context, id: str, title: str, description: str) -> None:
        Interface.update_journey(ctx, id, title, description)

    @journey.command(
        "add-condition",
        help="Add a condition to a journey, either by Guideline ID or by condition text",
    )
    @click.option("--id", type=str, metavar="ID", help="Journey ID", required=True)
    @click.option(
        "--guideline-id", type=str, metavar="GUIDELINE_ID", help="Guideline ID", required=False
    )
    @click.option("--condition", type=str, metavar="CONDITION", help="Condition", required=False)
    @click.pass_context
    def journey_add_condition(
        ctx: click.Context,
        id: str,
        condition: Optional[str],
        guideline_id: Optional[str],
    ) -> None:
        if not guideline_id and not condition:
            Interface.write_error("Either --condition-id or --condition must be provided")
            set_exit_status(1)
            raise FastExit()

        if guideline_id and condition:
            Interface.write_error("Only one of --condition-id or --condition can be provided")
            set_exit_status(1)
            raise FastExit()

        Interface.add_journey_condition(ctx, id, guideline_id, condition)

    @journey.command("remove-condition", help="Remove a condition from a journey")
    @click.option("--id", type=str, metavar="ID", help="Journey ID", required=True)
    @click.option("--condition", type=str, metavar="CONDITION", help="Condition", required=True)
    @click.pass_context
    def journey_remove_condition(ctx: click.Context, id: str, condition: str) -> None:
        Interface.remove_journey_condition(ctx, id, condition)

    @journey.command("tag", help="Tag a journey")
    @click.option("--id", type=str, metavar="ID", help="Journey ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def journey_add_tag(
        ctx: click.Context,
        id: str,
        tag: str,
    ) -> None:
        Interface.add_journey_tag(
            ctx=ctx,
            journey_id=id,
            tag=tag,
        )

    @journey.command("untag", help="Untag from a journey")
    @click.option("--id", type=str, metavar="ID", help="Journey ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def journey_untag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.remove_journey_tag(
            ctx=ctx,
            journey_id=id,
            tag=tag,
        )

    @journey.command("delete", help="Delete a journey")
    @click.option("--id", type=str, metavar="ID", help="Journey ID", required=True)
    @click.pass_context
    def journey_delete(ctx: click.Context, id: str) -> None:
        Interface.delete_journey(ctx, id)

    @cli.group(help="Manage capabilities")
    def capability() -> None:
        pass

    @capability.command("create", help="Create a capability")
    @click.option("--title", type=str, help="Capability title", required=True)
    @click.option("--description", type=str, help="Capability description", required=True)
    @click.option(
        "--query",
        type=str,
        help="Query for the capability. May be specified multiple times.",
        multiple=True,
        required=True,
    )
    @tag_option(multiple=True)
    @click.pass_context
    def capability_create(
        ctx: click.Context, title: str, description: str, query: tuple[str], tag: tuple[str]
    ) -> None:
        Interface.create_capability(ctx, title, description, list(query), list(tag))

    @capability.command(
        "update",
        help="Update a capability. If --query is provided, it will override all existing signals for this capability.",
    )
    @click.option("--id", type=str, metavar="ID", help="Capability ID", required=True)
    @click.option("--title", type=str, help="Capability title", required=False)
    @click.option("--description", type=str, help="Capability description", required=False)
    @click.option(
        "--signal",
        type=str,
        help="Signal for the capability. May be specified multiple times. If provided, overrides all existing signals.",
        multiple=True,
        required=False,
    )
    @click.pass_context
    def capability_update(
        ctx: click.Context,
        id: str,
        title: Optional[str],
        description: Optional[str],
        query: tuple[str],
    ) -> None:
        Interface.update_capability(ctx, id, title, description, list(query) if query else None)

    @capability.command("view", help="View a capability")
    @click.option("--id", type=str, metavar="ID", help="Capability ID", required=True)
    @click.pass_context
    def capability_view(ctx: click.Context, id: str) -> None:
        Interface.view_capability(ctx, id)

    @capability.command("list", help="List capabilities")
    @tag_option()
    @click.pass_context
    def capability_list(ctx: click.Context, tag: Optional[str]) -> None:
        Interface.list_capabilities(ctx, tag)

    @capability.command("tag", help="Tag a capability")
    @click.option("--id", type=str, metavar="ID", help="Capability ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def capability_add_tag(
        ctx: click.Context,
        id: str,
        tag: str,
    ) -> None:
        Interface.add_capability_tag(
            ctx=ctx,
            capability_id=id,
            tag=tag,
        )

    @capability.command("untag", help="Untag from a capability")
    @click.option("--id", type=str, metavar="ID", help="Capability ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def capability_untag(ctx: click.Context, id: str, tag: str) -> None:
        Interface.remove_capability_tag(
            ctx=ctx,
            capability_id=id,
            tag=tag,
        )

    @capability.command("delete", help="Delete a capability")
    @click.option("--id", type=str, metavar="ID", help="Capability ID", required=True)
    @click.pass_context
    def capability_delete(ctx: click.Context, id: str) -> None:
        Interface.delete_capability(ctx, id)

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

    cli(standalone_mode=False)


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
        except BaseException as exc:
            print(exc, file=sys.stderr)
            set_exit_status(1)

        sys.exit(get_exit_status())

    asyncio.run(wrapped_main())


if __name__ == "__main__":
    main()
