# Copyright 2026 Emcie Co Ltd.
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
    AgentGroupUpdateParams,
    Capability,
    CapabilityGroupUpdateParams,
    CannedResponse,
    CannedResponseField,
    ConsumptionOffsetsUpdateParams,
    ContextVariable,
    ContextVariableReadResult,
    ContextVariableValue,
    ContextVariableTagsUpdateParams,
    Customer,
    CustomerMetadataUpdateParams,
    CustomerGroupUpdateParams,
    Event,
    Journey,
    JourneyGraph,
    JourneyGroupUpdateParams,
    JourneyTriggerUpdateParams,
    Rule,
    Relationship,
    RelationshipKindDto,
    RulePayload,
    RuleContent,
    RuleToolAssociation,
    RuleToolAssociationUpdateParams,
    RuleGroupsUpdateParams,
    RuleWithRelationshipsAndToolAssociations,
    RuleMetadataUpdateParams,
    OpenApiServiceParams,
    Payload,
    SdkServiceParams,
    McpServiceParams,
    Service,
    Session,
    Term,
    TermGroupsUpdateParams,
    Tool,
    ToolId,
    Group,
)
from websocket import WebSocketConnectionClosedException, create_connection


INDENT = "  "


class FastExit(Exception):
    pass


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
    def _fetch_group_id(
        ctx: click.Context,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        if group.startswith("agent:"):
            agent_id = group.split(":")[1]
            if client.agents.retrieve(agent_id):
                return group
            else:
                raise Exception(f"Agent (id: {agent_id}) not found")

        if group.startswith("journey:"):
            journey_id = group.split(":")[1]
            if client.journeys.retrieve(journey_id):
                return group
            else:
                raise Exception(f"Journey (id: {journey_id}) not found")

        groups = client.groups.list()
        for t in groups:
            if t.name == group or t.id == group:
                return t.id

        raise Exception(f"Group ({group}) not found")

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
            if group_id := Actions._fetch_group_id(ctx, entity_id):
                return group_id, "group"

        with suppress(Exception):
            if ":" in entity_id and (
                tool_id := Actions._fetch_tool_id(
                    ctx,
                    ToolId(service_name=entity_id.split(":")[0], tool_name=entity_id.split(":")[1]),
                )
            ):
                return tool_id, "tool"

        client = cast(ParlantClient, ctx.obj.client)
        client.rules.retrieve(entity_id)
        return entity_id, "rule"

    @staticmethod
    def create_agent(
        ctx: click.Context,
        name: str,
        description: Optional[str],
        max_engine_iterations: Optional[int],
        composition_mode: Optional[str],
        groups: list[str],
    ) -> Agent:
        client = cast(ParlantClient, ctx.obj.client)

        return client.agents.create(
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
            composition_mode=composition_mode,
            groups=list(set([Actions._fetch_group_id(ctx, t) for t in groups])),
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
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        group_id = Actions._fetch_group_id(ctx, group)
        client.agents.update(
            agent_id=agent_id,
            groups=AgentGroupUpdateParams(add=[group_id]),
        )

        return group_id

    @staticmethod
    def remove_group(
        ctx: click.Context,
        agent_id: str,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        group_id = Actions._fetch_group_id(ctx, group)
        client.agents.update(
            agent_id=agent_id,
            groups=AgentGroupUpdateParams(remove=[group_id]),
        )

        return group_id

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

        return cast(
            list[Session],
            client.sessions.list(
                agent_id=agent_id,
                customer_id=customer_id,
            ),
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
        groups: list[str],
    ) -> Term:
        client = cast(ParlantClient, ctx.obj.client)

        return client.glossary.create_term(
            name=name,
            description=description,
            synonyms=synonyms,
            groups=list(set([Actions._fetch_group_id(ctx, t) for t in groups])),
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
        group: Optional[str] = None,
    ) -> list[Term]:
        client = cast(ParlantClient, ctx.obj.client)
        if group:
            return client.glossary.list_terms(group_id=Actions._fetch_group_id(ctx, group))
        else:
            return client.glossary.list_terms()

    @staticmethod
    def add_term_tag(
        ctx: click.Context,
        term_id: str,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        group_id = Actions._fetch_group_id(ctx, group)
        client.glossary.update_term(term_id, groups=TermGroupsUpdateParams(add=[group_id]))

        return group_id

    @staticmethod
    def remove_term_tag(
        ctx: click.Context,
        term_id: str,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        group_id = Actions._fetch_group_id(ctx, group)
        client.glossary.update_term(term_id, groups=TermGroupsUpdateParams(remove=[group_id]))

        return group_id

    @staticmethod
    def create_rule(
        ctx: click.Context,
        condition: str,
        action: Optional[str],
        tool_id: Optional[str],
        groups: list[str],
    ) -> RuleWithRelationshipsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)
        groups = list(set([Actions._fetch_group_id(ctx, t) for t in groups]))

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
                    kind="rule",
                    rule=RulePayload(
                        content=RuleContent(condition=condition),
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
            progress_task = progress.add_task("Evaluating rule\n", total=100)

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
                    assert invoice.data.rule
                    assert invoice.payload.rule

                    rule = client.rules.create(
                        condition=condition,
                        action=action if action else invoice.data.rule.action_proposition,
                        groups=groups,
                        metadata=invoice.data.rule.properties_proposition or {},
                    )

                    rule_with_relationships_and_associations = client.rules.update(
                        rule.id,
                        tool_associations=RuleToolAssociationUpdateParams(
                            add=tool_ids,
                        ),
                    )

                    return rule_with_relationships_and_associations

                elif evaluation_result.status == "failed":
                    raise ValueError(evaluation_result.error)

        if tool_id:
            tool_id_obj = Actions._fetch_tool_id(
                ctx, ToolId(service_name=tool_id.split(":")[0], tool_name=tool_id.split(":")[1])
            )

            rule_with_relationships_and_associations = client.rules.update(
                rule_id=rule.id,
                tool_associations=RuleToolAssociationUpdateParams(
                    add=[tool_id_obj],
                ),
            )

            return rule_with_relationships_and_associations

        return RuleWithRelationshipsAndToolAssociations(
            rule=rule,
            relationships=[],
            tool_associations=[],
        )

    @staticmethod
    def update_rule(
        ctx: click.Context,
        rule_id: str,
        condition: Optional[str] = None,
        action: Optional[str] = None,
    ) -> RuleWithRelationshipsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)

        return client.rules.update(rule_id, condition=condition, action=action)

    @staticmethod
    def delete_rule(
        ctx: click.Context,
        rule_id: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.rules.delete(rule_id)

    @staticmethod
    def view_rule(
        ctx: click.Context,
        rule_id: str,
    ) -> RuleWithRelationshipsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)
        return client.rules.retrieve(rule_id)

    @staticmethod
    def list_rules(
        ctx: click.Context,
        group: Optional[str],
    ) -> list[Rule]:
        client = cast(ParlantClient, ctx.obj.client)
        if group:
            return client.rules.list(group_id=Actions._fetch_group_id(ctx, group))
        else:
            return client.rules.list()

    @staticmethod
    def add_rule_tool_association(
        ctx: click.Context,
        rule_id: str,
        service_name: str,
        tool_name: str,
    ) -> RuleWithRelationshipsAndToolAssociations:
        client = cast(ParlantClient, ctx.obj.client)

        return client.rules.update(
            rule_id,
            tool_associations=RuleToolAssociationUpdateParams(
                add=[
                    ToolId(
                        service_name=service_name,
                        tool_name=tool_name,
                    ),
                ]
            ),
        )

    @staticmethod
    def remove_rule_tool_association(
        ctx: click.Context,
        rule_id: str,
        service_name: str,
        tool_name: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        rule_result = client.rules.retrieve(rule_id)
        associations = rule_result.tool_associations

        if association := next(
            (
                a
                for a in associations
                if a.tool_id.service_name == service_name and a.tool_id.tool_name == tool_name
            ),
            None,
        ):
            client.rules.update(
                rule_id,
                tool_associations=RuleToolAssociationUpdateParams(
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
            f"An association between {rule_id} and the tool {tool_name} from {service_name} was not found"
        )

    @staticmethod
    def enable_rule(
        ctx: click.Context,
        rule_ids: tuple[str],
    ) -> list[Rule]:
        client = cast(ParlantClient, ctx.obj.client)

        return [client.rules.update(rule_id, enabled=True).rule for rule_id in rule_ids]

    @staticmethod
    def disable_rule(
        ctx: click.Context,
        rule_ids: tuple[str],
    ) -> list[Rule]:
        client = cast(ParlantClient, ctx.obj.client)

        return [client.rules.update(rule_id, enabled=False).rule for rule_id in rule_ids]

    @staticmethod
    def add_rule_group(
        ctx: click.Context,
        rule_id: str,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        group_id = Actions._fetch_group_id(ctx, group)
        client.rules.update(rule_id, groups=RuleGroupsUpdateParams(add=[group_id]))

        return group_id

    @staticmethod
    def remove_rule_group(
        ctx: click.Context,
        rule_id: str,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        group_id = Actions._fetch_group_id(ctx, group)
        client.rules.update(rule_id, groups=RuleGroupsUpdateParams(remove=[group_id]))

        return group_id

    @staticmethod
    def set_rule_metadata(
        ctx: click.Context,
        rule_id: str,
        key: str,
        value: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.rules.update(
            rule_id,
            metadata=RuleMetadataUpdateParams(add={key: value}),
        )

    @staticmethod
    def unset_rule_metadata(
        ctx: click.Context,
        rule_id: str,
        key: str,
    ) -> None:
        client = cast(ParlantClient, ctx.obj.client)
        client.rules.update(
            rule_id,
            metadata=RuleMetadataUpdateParams(remove=[key]),
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
            source_rule=cast(str, source_id) if source_type == "rule" else None,
            source_group=cast(str, source_id) if source_type == "group" else None,
            source_tool=cast(ToolId, source_id) if source_type == "tool" else None,
            target_rule=cast(str, target_id) if target_type == "rule" else None,
            target_group=cast(str, target_id) if target_type == "group" else None,
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
                    rule_id=source_id if source_type == "rule" else None,
                    group_id=source_id if source_type == "group" else None,
                    tool_id=source_id if source_type == "tool" else None,
                    kind=kind,
                    indirect=False,
                )
                if (
                    (r.source_rule and source_id == r.source_rule.id)
                    or (r.source_group and source_id == r.source_group.id)
                    or (r.source_tool and source_id.split(":")[1] == r.source_tool.name)
                )
                and (
                    (r.target_rule and target_id == r.target_rule.id)
                    or (r.target_group and target_id == r.target_group.id)
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
        rule_id: Optional[str],
        group: Optional[str],
        tool_id: Optional[str],
        kind: Optional[RelationshipKindDto],
        indirect: Optional[bool],
    ) -> list[Relationship]:
        client = cast(ParlantClient, ctx.obj.client)

        group_id = Actions._fetch_group_id(ctx, group) if group else None
        if tool_id:
            Actions._fetch_tool_id(
                ctx, ToolId(service_name=tool_id.split(":")[0], tool_name=tool_id.split(":")[1])
            )

        return client.relationships.list(
            rule_id=rule_id,
            group_id=group_id,
            tool_id=tool_id,
            kind=kind,
            indirect=indirect,
        )

    @staticmethod
    def list_variables(
        ctx: click.Context,
        group: Optional[str],
    ) -> list[ContextVariable]:
        client = cast(ParlantClient, ctx.obj.client)
        if group:
            return client.context_variables.list(group_id=Actions._fetch_group_id(ctx, group))
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
        groups: list[str],
    ) -> ContextVariable:
        client = cast(ParlantClient, ctx.obj.client)

        return client.context_variables.create(
            name=name,
            description=description,
            tool_id=ToolId(service_name=service_name, tool_name=tool_name)
            if service_name and tool_name
            else None,
            freshness_rules=freshness_rules,
            groups=list(set([Actions._fetch_group_id(ctx, t) for t in groups])),
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

        if key.startswith("group:"):
            group_spec = key.split(":")[1]
            group_id = Actions._fetch_group_id(ctx, group_spec)
            key = f"group:{group_id}"

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

        if key.startswith("group:"):
            group_spec = key.split(":")[1]
            group_id = Actions._fetch_group_id(ctx, group_spec)
            key = f"group:{group_id}"

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
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        group_id = Actions._fetch_group_id(ctx, group)
        client.context_variables.update(
            variable_id, groups=ContextVariableTagsUpdateParams(add=[group_id])
        )
        return group_id

    @staticmethod
    def remove_variable_tag(
        ctx: click.Context,
        variable_id: str,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        group_id = Actions._fetch_group_id(ctx, group)
        client.context_variables.update(
            variable_id,
            groups=ContextVariableTagsUpdateParams(remove=[group_id]),
        )
        return group_id

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
            click.echo(
                click.style(
                    "Warning: OpenAPI tool services are deprecated and will be removed in a future version. "
                    "Please migrate to SDK tool services.",
                    fg="yellow",
                ),
                err=True,
            )
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
        return cast(list[Customer], client.customers.list())

    @staticmethod
    def create_customer(
        ctx: click.Context,
        name: str,
        groups: list[str],
    ) -> Customer:
        client = cast(ParlantClient, ctx.obj.client)
        return client.customers.create(
            name=name,
            metadata={},
            groups=list(set([Actions._fetch_group_id(ctx, t) for t in groups])),
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
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        group_id = Actions._fetch_group_id(ctx, group)
        client.customers.update(
            customer_id=customer_id,
            groups=CustomerGroupUpdateParams(add=[group_id]),
        )

        return group_id

    @staticmethod
    def remove_customer_tag(
        ctx: click.Context,
        customer_id: str,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        group_id = Actions._fetch_group_id(ctx, group)
        client.customers.update(
            customer_id=customer_id,
            groups=CustomerGroupUpdateParams(remove=[group_id]),
        )

        return group_id

    @staticmethod
    def list_groups(ctx: click.Context) -> list[Group]:
        client = cast(ParlantClient, ctx.obj.client)
        return client.groups.list()

    @staticmethod
    def create_group(
        ctx: click.Context,
        name: str,
    ) -> Group:
        client = cast(ParlantClient, ctx.obj.client)
        return client.groups.create(name=name)

    @staticmethod
    def view_tag(
        ctx: click.Context,
        group: str,
    ) -> Group:
        group_id = Actions._fetch_group_id(ctx, group)

        client = cast(ParlantClient, ctx.obj.client)
        return client.groups.retrieve(group_id=group_id)

    @staticmethod
    def update_group(
        ctx: click.Context,
        group: str,
        name: str,
    ) -> Group:
        client = cast(ParlantClient, ctx.obj.client)
        return client.groups.update(group_id=Actions._fetch_group_id(ctx, group), name=name)

    @staticmethod
    def delete_group(
        ctx: click.Context,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        group_id = Actions._fetch_group_id(ctx, group)
        client.groups.delete(group_id=group_id)

        return group_id

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
        group_ids = {group.name: group.id for group in client.groups.list()}

        for canned_response_data in data.get("canned_responses", []):
            value = canned_response_data["value"]
            assert value

            fields = [
                CannedResponseField(**canned_response_field)
                for canned_response_field in canned_response_data.get("fields", [])
            ]

            group_names = canned_response_data.get("groups", [])

            signals = canned_response_data.get("signals", [])

            canned_response = client.canned_responses.create(
                value=value,
                fields=fields,
                groups=[
                    group_ids[group_name] for group_name in group_names if group_name in group_ids
                ]
                or None,
                signals=signals,
            )

            canned_responses.append(canned_response)

        return canned_responses

    @staticmethod
    def list_journeys(
        ctx: click.Context,
        group: Optional[str] = None,
    ) -> list[Journey]:
        client = cast(ParlantClient, ctx.obj.client)
        if group:
            return client.journeys.list(group_id=Actions._fetch_group_id(ctx, group))
        else:
            return client.journeys.list()

    @staticmethod
    def create_journey(
        ctx: click.Context,
        title: str,
        description: str,
        triggers: list[str],
        groups: list[str],
    ) -> Journey:
        client = cast(ParlantClient, ctx.obj.client)

        journey = client.journeys.create(
            title=title,
            description=description,
            triggers=triggers,
            groups=groups,
        )

        return journey

    @staticmethod
    def view_journey(
        ctx: click.Context,
        journey_id: str,
    ) -> JourneyGraph:
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
    def add_journey_trigger(
        ctx: click.Context,
        journey_id: str,
        rule_id: Optional[str],
        trigger: Optional[str],
    ) -> Journey:
        client = cast(ParlantClient, ctx.obj.client)

        rule_id = (
            rule_id
            or client.rules.create(
                condition=cast(str, trigger),
                metadata={"journeys": [journey_id]},
            ).id
        )

        return client.journeys.update(
            journey_id=journey_id,
            triggers=JourneyTriggerUpdateParams(add=[rule_id]),
        )

    @staticmethod
    def remove_journey_trigger(
        ctx: click.Context,
        journey_id: str,
        rule_id: str,
    ) -> Journey:
        client = cast(ParlantClient, ctx.obj.client)

        return client.journeys.update(
            journey_id=journey_id,
            triggers=JourneyTriggerUpdateParams(remove=[rule_id]),
        )

    @staticmethod
    def add_journey_group(
        ctx: click.Context,
        journey_id: str,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        group_id = Actions._fetch_group_id(ctx, group)
        client.journeys.update(
            journey_id=journey_id, groups=JourneyGroupUpdateParams(add=[group_id])
        )

        return group_id

    @staticmethod
    def remove_journey_group(
        ctx: click.Context,
        journey_id: str,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)
        group_id = Actions._fetch_group_id(ctx, group)
        client.journeys.update(
            journey_id=journey_id, groups=JourneyGroupUpdateParams(remove=[group_id])
        )

        return group_id

    @staticmethod
    def create_capability(
        ctx: click.Context,
        title: str,
        description: str,
        signals: list[str],
        groups: list[str],
    ) -> Capability:
        client = cast(ParlantClient, ctx.obj.client)
        groups = list(set([Actions._fetch_group_id(ctx, t) for t in groups]))

        return client.capabilities.create(
            title=title,
            description=description,
            signals=signals,
            groups=groups,
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
        group: Optional[str],
    ) -> list[Capability]:
        client = cast(ParlantClient, ctx.obj.client)

        if group:
            return client.capabilities.list(group_id=Actions._fetch_group_id(ctx, group))
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
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        group_id = Actions._fetch_group_id(ctx, group)
        client.capabilities.update(
            capability_id, groups=CapabilityGroupUpdateParams(add=[group_id])
        )

        return group_id

    @staticmethod
    def remove_capability_tag(
        ctx: click.Context,
        capability_id: str,
        group: str,
    ) -> str:
        client = cast(ParlantClient, ctx.obj.client)

        group_id = Actions._fetch_group_id(ctx, group)
        client.capabilities.update(
            capability_id,
            groups=CapabilityGroupUpdateParams(remove=[group_id]),
        )

        return group_id

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
                "Groups": ", ".join(a.groups or []),
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
        groups: list[str],
    ) -> None:
        try:
            agent = Actions.create_agent(
                ctx,
                name,
                description,
                max_engine_iterations,
                composition_mode,
                groups,
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
    def add_tag(ctx: click.Context, agent_id: str, group: str) -> None:
        try:
            group_id = Actions.add_tag(ctx, agent_id, group)
            Interface._write_success(f"Grouped agent (id: {agent_id}, group_id: {group_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_group(ctx: click.Context, agent_id: str, group: str) -> None:
        try:
            group_id = Actions.remove_group(ctx, agent_id, group)
            Interface._write_success(f"Untagged agent (id: {agent_id}, group_id: {group_id})")
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
                "Trace ID": e.trace_id,
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
                "Groups": ", ".join(term.groups),
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
        groups: list[str],
    ) -> None:
        term = Actions.create_term(
            ctx,
            name,
            description,
            synonyms,
            groups=groups,
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
        group: Optional[str],
    ) -> None:
        terms = Actions.list_terms(ctx, group)

        if not terms:
            rich.print(Text("No data available", style="bold yellow"))
            return

        Interface._render_glossary(terms)

    @staticmethod
    def add_term_tag(
        ctx: click.Context,
        term_id: str,
        group: str,
    ) -> None:
        group_id = Actions.add_term_tag(ctx, term_id, group)
        Interface._write_success(f"Added group (id: {group_id}) to term (id: {term_id})")

    @staticmethod
    def remove_term_tag(
        ctx: click.Context,
        term_id: str,
        group: str,
    ) -> None:
        group_id = Actions.remove_term_tag(ctx, term_id, group)
        Interface._write_success(f"Removed group (id: {group_id}) from term (id: {term_id})")

    @staticmethod
    def _render_rules(rules: list[Rule]) -> None:
        rule_items: list[dict[str, Any]] = [
            {
                "ID": rule.id,
                "Condition": rule.condition,
                "Action": (
                    rule.action
                    if rule.action
                    else f"Activate journey(s): {', '.join(group.split('journey:')[1] for group in rule.groups if group.startswith('journey:'))}"
                    if any(group for group in rule.groups if group.startswith("journey:"))
                    else "None"
                ),
                "Enabled": rule.enabled,
                "Groups": ", ".join(rule.groups),
                "Metadata": ", ".join([f"{k}: {v}" for k, v in rule.metadata.items()])
                if rule.metadata
                else "",
            }
            for rule in rules
        ]

        Interface._print_table(rule_items)

    @staticmethod
    def _render_relationships(
        entity: Rule | Group | Tool | None,
        relationships: list[Relationship],
        include_indirect: bool,
    ) -> None:
        def to_direct_relationship_item(rel: Relationship) -> dict[str, str]:
            result: dict[str, str] = {
                "Relationship ID": rel.id,
                "Kind": rel.kind,
            }

            if rel.source_rule:
                result.update(
                    {
                        "Source ID": rel.source_rule.id,
                        "Source Type": "Rule",
                        "Source Condition": rel.source_rule.condition,
                        "Source Action": rel.source_rule.action or "",
                    }
                )
            elif rel.source_group:
                assert rel.source_group is not None
                result.update(
                    {
                        "Source ID": rel.source_group.id,
                        "Source Type": "Group",
                        "Source Name": rel.source_group.name,
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
            if rel.target_rule:
                result.update(
                    {
                        "Target ID": rel.target_rule.id,
                        "Target Type": "Rule",
                        "Target Condition": rel.target_rule.condition,
                        "Target Action": rel.target_rule.action or "",
                    }
                )
            elif rel.target_group:
                assert rel.target_group is not None
                result.update(
                    {
                        "Target ID": rel.target_group.id,
                        "Target Type": "Group",
                        "Target Name": rel.target_group.name,
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

            if rel.source_rule:
                result.update(
                    {
                        "Source ID": rel.source_rule.id,
                        "Source Type": "Rule",
                        "Source Condition": rel.source_rule.condition,
                        "Source Action": rel.source_rule.action or "",
                    }
                )
            elif rel.source_group:
                result.update(
                    {
                        "Source ID": rel.source_group.id,
                        "Source Type": "Group",
                        "Source Name": rel.source_group.name,
                    }
                )
            elif rel.source_tool:
                result.update(
                    {
                        "Source Type": "Tool",
                        "Source Name": rel.source_tool.name,
                    }
                )
            if rel.target_rule:
                result.update(
                    {
                        "Target ID": rel.target_rule.id,
                        "Target Type": "Rule",
                        "Target Condition": rel.target_rule.condition,
                        "Target Action": rel.target_rule.action or "",
                    }
                )
            elif rel.target_group:
                result.update(
                    {
                        "Target ID": rel.target_group.id,
                        "Target Type": "Group",
                        "Target Name": rel.target_group.name,
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
                    r.source_rule,
                    r.target_rule,
                    r.source_group,
                    r.target_group,
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
    def create_rule(
        ctx: click.Context,
        condition: str,
        action: Optional[str],
        tool_id: Optional[str],
        groups: tuple[str],
    ) -> None:
        try:
            rule_with_relationships_and_associations = Actions.create_rule(
                ctx,
                condition,
                action,
                tool_id,
                groups=list(groups),
            )

            Interface._write_success(
                f"Added rule (id: {rule_with_relationships_and_associations.rule.id})"
            )
            Interface._render_rules([rule_with_relationships_and_associations.rule])
            Interface._render_relationships(
                rule_with_relationships_and_associations.rule,
                rule_with_relationships_and_associations.relationships,
                include_indirect=False,
            )
            Interface._render_rule_tool_associations(
                rule_with_relationships_and_associations.tool_associations
            )

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def update_rule(
        ctx: click.Context,
        rule_id: str,
        condition: str,
        action: str,
    ) -> None:
        try:
            rule_with_relationships_and_associations = Actions.update_rule(
                ctx,
                rule_id,
                condition=condition,
                action=action,
            )

            rule = rule_with_relationships_and_associations.rule
            Interface._write_success(f"Updated rule (id: {rule.id})")
            Interface._render_relationships(
                rule_with_relationships_and_associations.rule,
                rule_with_relationships_and_associations.relationships,
                include_indirect=False,
            )
            Interface._render_rule_tool_associations(
                rule_with_relationships_and_associations.tool_associations
            )

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_rule(
        ctx: click.Context,
        rule_id: str,
    ) -> None:
        try:
            Actions.delete_rule(ctx, rule_id)

            Interface._write_success(f"Removed rule (id: {rule_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_rule(
        ctx: click.Context,
        rule_id: str,
    ) -> None:
        try:
            rule_with_relationships_and_associations = Actions.view_rule(ctx, rule_id)

            Interface._render_rules([rule_with_relationships_and_associations.rule])
            Interface._render_relationships(
                rule_with_relationships_and_associations.rule,
                rule_with_relationships_and_associations.relationships,
                include_indirect=True,
            )
            Interface._render_rule_tool_associations(
                rule_with_relationships_and_associations.tool_associations
            )

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def list_rules(
        ctx: click.Context,
        group: Optional[str],
        hide_disabled: bool,
    ) -> None:
        try:
            rules = Actions.list_rules(ctx, group)

            rules_to_render = sorted(
                [g for g in rules if g.enabled or not hide_disabled],
                key=lambda g: g.enabled or False,
                reverse=True,
            )

            if not rules_to_render:
                rich.print(Text("No data available", style="bold yellow"))
                return

            Interface._render_rules(rules_to_render)

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_rule_tool_associations(
        associations: list[RuleToolAssociation],
    ) -> None:
        if associations:
            association_items = [
                {
                    "Association ID": a.id,
                    "Rule ID": a.rule_id,
                    "Service Name": a.tool_id.service_name,
                    "Tool Name": a.tool_id.tool_name,
                }
                for a in associations
            ]

            Interface._print_table(association_items)

    @staticmethod
    def add_rule_tool_association(
        ctx: click.Context,
        rule_id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        try:
            rule = Actions.add_rule_tool_association(ctx, rule_id, service_name, tool_name)

            Interface._write_success(
                f"Enabled tool '{tool_name}' from service '{service_name}' for rule '{rule_id}'"
            )
            Interface._render_rule_tool_associations(rule.tool_associations)

        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_rule_tool_association(
        ctx: click.Context,
        rule_id: str,
        service_name: str,
        tool_name: str,
    ) -> None:
        try:
            association_id = Actions.remove_rule_tool_association(
                ctx, rule_id, service_name, tool_name
            )

            Interface._write_success(f"Removed tool association (id: {association_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def enable_rule(
        ctx: click.Context,
        rule_ids: tuple[str],
    ) -> None:
        try:
            rules = Actions.enable_rule(ctx, rule_ids)

            Interface._write_success(f"Enabled rules (ids: {', '.join(rule_ids)})")

            Interface._render_rules(rules)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def disable_rule(
        ctx: click.Context,
        rule_ids: tuple[str],
    ) -> None:
        try:
            rules = Actions.disable_rule(ctx, rule_ids)

            Interface._write_success(f"Disabled rules (ids: {', '.join(rule_ids)})")

            Interface._render_rules(rules)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_rule_group(
        ctx: click.Context,
        rule_id: str,
        group: str,
    ) -> None:
        try:
            group_id = Actions.add_rule_group(ctx, rule_id, group)
            Interface._write_success(f"Added group (id: {group_id}) to rule (id: {rule_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_rule_group(
        ctx: click.Context,
        rule_id: str,
        group: str,
    ) -> None:
        try:
            group_id = Actions.remove_rule_group(ctx, rule_id, group)
            Interface._write_success(f"Removed group (id: {group_id}) from rule (id: {rule_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def set_rule_metadata(
        ctx: click.Context,
        rule_id: str,
        key: str,
        value: str,
    ) -> None:
        try:
            Actions.set_rule_metadata(ctx, rule_id, key, value)
            Interface._write_success(
                f"Added metadata (key: {key}, value: {value}) to rule (id: {rule_id})"
            )
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def unset_rule_metadata(
        ctx: click.Context,
        rule_id: str,
        key: str,
    ) -> None:
        try:
            Actions.unset_rule_metadata(ctx, rule_id, key)
            Interface._write_success(f"Removed metadata (key: {key}) from rule (id: {rule_id})")
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
        rule_id: Optional[str],
        group: Optional[str],
        tool_id: Optional[str],
        kind: Optional[RelationshipKindDto],
        indirect: Optional[bool],
    ) -> None:
        try:
            relationships = Actions.list_relationships(
                ctx,
                rule_id=rule_id,
                group=group,
                tool_id=tool_id,
                kind=kind,
                indirect=indirect,
            )

            if not relationships:
                rich.print(Text("No data available", style="bold yellow"))
                return

            entity: Rule | Group | Tool | None = None
            if rule_id:
                entity = Actions.view_rule(ctx, rule_id).rule
            elif group:
                entity = Actions.view_tag(ctx, group)
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
                "Groups": ", ".join(variable.groups or []),
            }
            for variable in variables
        ]

        Interface._print_table(variable_items)

    @staticmethod
    def list_variables(
        ctx: click.Context,
        group: Optional[str],
    ) -> None:
        variables = Actions.list_variables(ctx, group)

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
        groups: list[str],
    ) -> None:
        variable = Actions.create_variable(
            ctx,
            name,
            description,
            service_name,
            tool_name,
            freshness_rules,
            groups=groups,
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
    def add_variable_tag(ctx: click.Context, variable_id: str, group: str) -> None:
        try:
            group_id = Actions.add_variable_tag(ctx, variable_id, group)
            Interface._write_success(
                f"Added group (id: {group_id}) to variable (id: {variable_id})"
            )
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_variable_tag(ctx: click.Context, variable_id: str, group: str) -> None:
        try:
            group_id = Actions.remove_variable_tag(ctx, variable_id, group)
            Interface._write_success(
                f"Removed group (id: {group_id}) from variable (id: {variable_id})"
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
                "Groups": ", ".join(customer.groups),
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
        groups: list[str],
    ) -> None:
        try:
            customer = Actions.create_customer(
                ctx,
                name,
                groups,
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
        group: str,
    ) -> None:
        try:
            group_id = Actions.add_customer_tag(ctx, customer_id, group)
            Interface._write_success(f"Grouped customer (id: {customer_id}, group_id: {group_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_customer_tag(
        ctx: click.Context,
        customer_id: str,
        group: str,
    ) -> None:
        try:
            group_id = Actions.remove_customer_tag(ctx, customer_id, group)
            Interface._write_success(f"Untagged customer (id: {customer_id}, group_id: {group_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def _render_tags(groups: list[Group]) -> None:
        tag_items: list[dict[str, Any]] = [
            {
                "ID": group.id,
                "Name": group.name,
            }
            for group in groups
        ]

        Interface._print_table(tag_items)

    @staticmethod
    def list_groups(ctx: click.Context) -> None:
        try:
            groups = Actions.list_groups(ctx)
            if not groups:
                rich.print("No groups found.")
                return

            Interface._render_tags(groups)
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def create_group(ctx: click.Context, name: str) -> None:
        try:
            group = Actions.create_group(ctx, name=name)
            Interface._write_success(f"Added group (id: {group.id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def view_tag(ctx: click.Context, group: str) -> None:
        try:
            tag_dto = Actions.view_tag(ctx, group)
            Interface._render_tags([tag_dto])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def update_group(ctx: click.Context, group: str, name: str) -> None:
        try:
            tag_dto = Actions.update_group(ctx, group=group, name=name)
            Interface._write_success(f"Updated group (id: {tag_dto.id})")

            Interface._render_tags([tag_dto])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def delete_group(ctx: click.Context, group: str) -> None:
        try:
            group_id = Actions.delete_group(ctx, group)
            Interface._write_success(f"Removed group (id: {group_id})")
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
                "Groups": ", ".join(f.groups),
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
                "Trigger Rule IDs": ", ".join(journey.triggers),
                "Groups": ", ".join(journey.groups or []),
            }
            for journey in journeys
        ]

        Interface._print_table(journey_items)

    @staticmethod
    def list_journeys(
        ctx: click.Context,
        group: Optional[str],
    ) -> None:
        try:
            journeys = Actions.list_journeys(ctx, group)

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
        triggers: list[str],
        groups: list[str],
    ) -> None:
        try:
            journey = Actions.create_journey(ctx, title, description, triggers, groups)
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
    def add_journey_trigger(
        ctx: click.Context,
        journey_id: str,
        rule_id: Optional[str],
        trigger: Optional[str],
    ) -> None:
        try:
            journey = Actions.add_journey_trigger(ctx, journey_id, rule_id, trigger)
            Interface._write_success(f"Added trigger to journey (id: {journey.id})")
            Interface._render_journeys([journey])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_journey_trigger(
        ctx: click.Context,
        journey_id: str,
        rule_id: str,
    ) -> None:
        try:
            journey = Actions.remove_journey_trigger(ctx, journey_id, rule_id)
            Interface._write_success(f"Removed trigger from journey (id: {journey.id})")
            Interface._render_journeys([journey])
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def add_journey_group(
        ctx: click.Context,
        journey_id: str,
        group: str,
    ) -> None:
        try:
            group_id = Actions.add_journey_group(ctx, journey_id, group)
            Interface._write_success(f"Added group (id: {group_id}) to journey (id: {journey_id})")
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_journey_group(
        ctx: click.Context,
        journey_id: str,
        group: str,
    ) -> None:
        try:
            group_id = Actions.remove_journey_group(ctx, journey_id, group)
            Interface._write_success(
                f"Removed group (id: {group_id}) from journey (id: {journey_id})"
            )
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
                "Groups": ", ".join(c.groups or []),
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
        groups: list[str],
    ) -> None:
        try:
            capability = Actions.create_capability(ctx, title, description, signals, groups)

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
    def list_capabilities(ctx: click.Context, group: Optional[str]) -> None:
        try:
            capabilities = Actions.list_capabilities(ctx, group)

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
        group: str,
    ) -> None:
        try:
            group_id = Actions.add_capability_tag(ctx, capability_id, group)

            Interface._write_success(
                f"Added group (id: {group_id}) to capability (id: {capability_id})"
            )
        except Exception as e:
            Interface.write_error(f"Error: {type(e).__name__}: {e}")
            set_exit_status(1)

    @staticmethod
    def remove_capability_tag(
        ctx: click.Context,
        capability_id: str,
        group: str,
    ) -> None:
        try:
            group_id = Actions.remove_capability_tag(ctx, capability_id, group)

            Interface._write_success(
                f"Removed group (id: {group_id}) from capability (id: {capability_id})"
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
                trace_id = log.get("trace_id", "")
                rich.print(f"[{level}] [{trace_id}] {message}")
        except Exception as e:
            Interface.write_error(f"Error while streaming logs: {e}")
            set_exit_status(1)


def tag_option(
    required: bool = False,
    multiple: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        return click.option(
            "--group",
            type=str,
            metavar="GROUP_NAME | GROUP_ID",
            help="Group name or ID. May be specified multiple times.",
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
        group: tuple[str],
    ) -> None:
        if composition_mode:
            composition_mode = composition_mode.replace("-", "_")

        Interface.create_agent(
            ctx=ctx,
            name=name,
            description=description,
            max_engine_iterations=max_engine_iterations,
            composition_mode=composition_mode,
            groups=list(group),
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

    @agent.command("group", help="Group an agent")
    @click.option("--id", type=str, metavar="ID", help="Agent ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def agent_group(ctx: click.Context, id: str, group: str) -> None:
        Interface.add_tag(ctx, id, group)

    @agent.command("untag", help="Untag an agent")
    @click.option("--id", type=str, metavar="ID", help="Agent ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def agent_remove_group(ctx: click.Context, id: str, group: str) -> None:
        Interface.remove_group(ctx, id, group)

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
        group: tuple[str],
    ) -> None:
        Interface.create_term(
            ctx,
            name,
            description,
            (synonyms or "").split(","),
            list(group),
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
        group: Optional[str],
    ) -> None:
        Interface.list_terms(ctx, group)

    @glossary.command("group", help="Group a term")
    @click.option("--id", type=str, metavar="ID", help="Term ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def glossary_tag(
        ctx: click.Context,
        id: str,
        group: str,
    ) -> None:
        Interface.add_term_tag(
            ctx=ctx,
            term_id=id,
            group=group,
        )

    @glossary.command("untag", help="Untag from a term")
    @click.option("--id", type=str, metavar="ID", help="Term ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def glossary_untag(
        ctx: click.Context,
        id: str,
        group: str,
    ) -> None:
        Interface.remove_term_tag(
            ctx=ctx,
            term_id=id,
            group=group,
        )

    @cli.group(help="Manage an agent's rules")
    def rule() -> None:
        pass

    @rule.command("create", help="Create a rule")
    @click.option(
        "--condition",
        type=str,
        help="A statement describing when the rule should apply",
        required=True,
    )
    @click.option(
        "--action",
        type=str,
        help="The instruction to perform when the rule applies",
        required=False,
    )
    @click.option(
        "--tool-id",
        type=str,
        help="The ID of the tool to associate with the rule, in the format service_name:tool_name",
        required=False,
    )
    @tag_option(multiple=True)
    @click.pass_context
    def rule_create(
        ctx: click.Context,
        condition: str,
        action: Optional[str],
        tool_id: Optional[str],
        group: tuple[str],
    ) -> None:
        Interface.create_rule(
            ctx=ctx,
            condition=condition,
            action=action,
            tool_id=tool_id,
            groups=group,
        )

    @rule.command("update", help="Update a rule")
    @click.option("--id", type=str, metavar="ID", help="Rule ID", required=True)
    @click.option(
        "--condition",
        type=str,
        help="A statement describing when the rule should apply",
        required=False,
    )
    @click.option(
        "--action",
        type=str,
        help="The instruction to perform when the rule applies",
        required=False,
    )
    @click.pass_context
    def rule_update(
        ctx: click.Context,
        id: str,
        condition: str,
        action: str,
    ) -> None:
        if not (condition or action):
            Interface.write_error("At least one of --condition or --action must be specified")
            set_exit_status(1)
            raise FastExit()

        Interface.update_rule(
            ctx=ctx,
            rule_id=id,
            condition=condition,
            action=action,
        )

    @rule.command("delete", help="Delete a rule")
    @click.option("--id", type=str, metavar="ID", help="Rule ID", required=True)
    @click.pass_context
    def rule_delete(
        ctx: click.Context,
        id: str,
    ) -> None:
        Interface.delete_rule(ctx, id)

    @rule.command("view", help="View a rule")
    @click.option("--id", type=str, metavar="ID", help="Rule ID", required=True)
    @click.pass_context
    def rule_view(
        ctx: click.Context,
        id: str,
    ) -> None:
        Interface.view_rule(ctx, id)

    @rule.command("list", help="List rules")
    @tag_option()
    @click.option(
        "--hide-disabled",
        type=bool,
        show_default=True,
        default=False,
        help="Hide disabled rules",
    )
    @click.pass_context
    def rule_list(
        ctx: click.Context,
        group: Optional[str],
        hide_disabled: bool,
    ) -> None:
        Interface.list_rules(ctx, group, hide_disabled)

    @rule.command("tool-enable", help="Allow a rule to make use of a tool")
    @click.option("--id", type=str, metavar="ID", help="Rule ID", required=False)
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
    def rule_enable_tool(
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

        Interface.add_rule_tool_association(
            ctx=ctx,
            rule_id=id,
            service_name=service_name,
            tool_name=tool_name,
        )

    @rule.command("tool-disable", help="Disallow a rule to make use of a tool")
    @click.option("--id", type=str, metavar="ID", help="Rule ID", required=True)
    @click.option(
        "--service",
        type=str,
        metavar="NAME",
        help="The name of the tool service containing the tool",
        required=True,
    )
    @click.option("--tool", type=str, metavar="NAME", help="Tool name", required=True)
    @click.pass_context
    def rule_disable_tool(
        ctx: click.Context,
        id: str,
        service: str,
        tool: str,
    ) -> None:
        Interface.remove_rule_tool_association(
            ctx=ctx,
            rule_id=id,
            service_name=service,
            tool_name=tool,
        )

    @rule.command("enable", help="Enable a rule")
    @click.option(
        "--id",
        "ids",
        type=str,
        metavar="ID",
        help="Rule ID, May be specified multiple times.",
        required=True,
        multiple=True,
    )
    @click.pass_context
    def rule_enable(
        ctx: click.Context,
        ids: tuple[str],
    ) -> None:
        Interface.enable_rule(
            ctx=ctx,
            rule_ids=ids,
        )

    @rule.command("disable", help="Disable a rule")
    @click.option(
        "--id",
        "ids",
        type=str,
        metavar="ID",
        help="Rule ID, May be specified multiple times.",
        required=True,
        multiple=True,
    )
    @click.pass_context
    def rule_disable(
        ctx: click.Context,
        ids: tuple[str],
    ) -> None:
        Interface.disable_rule(
            ctx=ctx,
            rule_ids=ids,
        )

    @rule.command("group", help="Group a rule")
    @click.option("--id", type=str, metavar="ID", help="Rule ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def rule_group(
        ctx: click.Context,
        id: str,
        group: str,
    ) -> None:
        Interface.add_rule_group(
            ctx=ctx,
            rule_id=id,
            group=group,
        )

    @rule.command("untag", help="Untag from a rule")
    @click.option("--id", type=str, metavar="ID", help="Rule ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def rule_untag(
        ctx: click.Context,
        id: str,
        group: str,
    ) -> None:
        Interface.remove_rule_group(
            ctx=ctx,
            rule_id=id,
            group=group,
        )

    @rule.command("set", help="Set metadata for a rule using a key and value")
    @click.option("--id", type=str, metavar="ID", help="Rule ID", required=True)
    @click.option("--key", type=str, metavar="KEY", help="Key", required=True)
    @click.option("--value", type=str, metavar="VALUE", help="Value", required=True)
    @click.pass_context
    def rule_set(ctx: click.Context, id: str, key: str, value: str) -> None:
        Interface.set_rule_metadata(ctx, id, key, value)

    @rule.command("unset", help="Remove metadata for a rule using a key")
    @click.option("--id", type=str, metavar="ID", help="Rule ID", required=True)
    @click.option("--key", type=str, metavar="KEY", help="Key", required=True)
    @click.pass_context
    def rule_unset(ctx: click.Context, id: str, key: str) -> None:
        Interface.unset_rule_metadata(ctx, id, key)

    @cli.group(help="Manage relationships")
    def relationship() -> None:
        pass

    @relationship.command("create", help="Create a relationship")
    @click.option(
        "--source",
        type=str,
        metavar="GROUP_NAME | GROUP_ID | RULE_ID | TOOL_ID",
        help="Source group or rule ID or tool ID",
        required=True,
    )
    @click.option(
        "--target",
        type=str,
        metavar="GROUP_NAME | GROUP_ID | RULE_ID | TOOL_ID",
        help="Target group or rule ID or tool ID",
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

    @relationship.command("delete", help="Delete a relationship between two rules")
    @click.option("--id", type=str, metavar="ID", help="Relationship ID")
    @click.option(
        "--source",
        type=str,
        metavar="RULE_ID",
        help="Source of the relationship",
    )
    @click.option(
        "--target",
        type=str,
        metavar="GROUP_NAME | GROUP_ID | RULE_ID",
        help="Target group or rule ID",
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
        "--rule-id",
        type=str,
        metavar="RULE_ID",
        help="Rule ID",
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
        rule_id: Optional[str],
        group: Optional[str],
        tool: Optional[str],
        kind: Optional[RelationshipKindDto],
        indirect: Optional[bool],
    ) -> None:
        if rule_id and group:
            Interface.write_error("Either --rule-id or --group must be provided, not both")
            set_exit_status(1)
            raise FastExit()

        Interface.list_relationships(ctx, rule_id, group, tool, kind, indirect)

    @cli.group(help="Manage an agent's context variables")
    def variable() -> None:
        pass

    @variable.command("list", help="List variables")
    @tag_option()
    @click.pass_context
    def variable_list(
        ctx: click.Context,
        group: Optional[str],
    ) -> None:
        Interface.list_variables(
            ctx=ctx,
            group=group,
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
        group: tuple[str],
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
            groups=list(group),
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
        help='The key (e.g. <CUSTOMER_ID> or "group:<GROUP_ID>" or "DEFAULT" to set a default value)',
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
        help='The key (e.g. <CUSTOMER_ID> or "group:<GROUP_ID>" or "DEFAULT" to set a default value)',
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
        help='The key (e.g. <CUSTOMER_ID> or "group:<GROUP_ID>" or "DEFAULT" to set a default value)',
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

    @variable.command("group", help="Group a variable")
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def variable_tag(ctx: click.Context, id: str, group: str) -> None:
        Interface.add_variable_tag(ctx, id, group)

    @variable.command("untag", help="Untag a variable")
    @click.option("--id", type=str, metavar="ID", help="Variable ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def variable_untag(ctx: click.Context, id: str, group: str) -> None:
        Interface.remove_variable_tag(ctx, id, group)

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
        group: tuple[str],
    ) -> None:
        Interface.create_customer(
            ctx,
            name,
            list(group),
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

    @customer.command("group", help="Group a customer")
    @click.option("--id", type=str, metavar="ID", help="Customer ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def customer_tag(ctx: click.Context, id: str, group: str) -> None:
        Interface.add_customer_tag(ctx, id, group)

    @customer.command("untag", help="Untag a customer")
    @click.option("--id", type=str, metavar="ID", help="Customer ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def customer_untag(ctx: click.Context, id: str, group: str) -> None:
        Interface.remove_customer_tag(ctx, id, group)

    @cli.group(help="Manage groups")
    def group() -> None:
        """Group of commands to manage groups."""

    @group.command("list", help="List groups")
    @click.pass_context
    def tag_list(ctx: click.Context) -> None:
        Interface.list_groups(ctx)

    @group.command("create", help="Create a group")
    @click.option("--name", type=str, metavar="NAME", help="Group name", required=True)
    @click.pass_context
    def tag_create(ctx: click.Context, name: str) -> None:
        Interface.create_group(ctx, name)

    @group.command("view", help="View a group")
    @tag_option(required=True)
    @click.pass_context
    def tag_view(ctx: click.Context, group: str) -> None:
        Interface.view_tag(ctx, group)

    @group.command("update", help="Update a group")
    @click.option("--id", type=str, metavar="ID", help="Group ID", required=True)
    @click.option("--name", type=str, metavar="NAME", help="Group name", required=True)
    @click.pass_context
    def tag_update(ctx: click.Context, id: str, name: str) -> None:
        Interface.update_group(ctx, id, name)

    @group.command("delete", help="Delete a group")
    @tag_option(required=True)
    @click.pass_context
    def tag_delete(ctx: click.Context, group: str) -> None:
        Interface.delete_group(ctx, group)

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
        group: Optional[str],
    ) -> None:
        Interface.list_journeys(ctx, group)

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
        "--trigger",
        type=str,
        metavar="TRIGGER",
        help="Journey triggers",
        multiple=True,
        required=True,
    )
    @tag_option(multiple=True)
    @click.pass_context
    def journey_create(
        ctx: click.Context,
        title: str,
        description: str,
        trigger: tuple[str],
        group: tuple[str],
    ) -> None:
        Interface.create_journey(
            ctx=ctx,
            title=title,
            description=description,
            triggers=list(trigger),
            groups=list(group),
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
        "add-trigger",
        help="Add a trigger to a journey, either by Rule ID or by trigger text",
    )
    @click.option("--id", type=str, metavar="ID", help="Journey ID", required=True)
    @click.option("--rule-id", type=str, metavar="RULE_ID", help="Rule ID", required=False)
    @click.option("--trigger", type=str, metavar="TRIGGER", help="Trigger", required=False)
    @click.pass_context
    def journey_add_trigger(
        ctx: click.Context,
        id: str,
        trigger: Optional[str],
        rule_id: Optional[str],
    ) -> None:
        if not rule_id and not trigger:
            Interface.write_error("Either --rule-id or --trigger must be provided")
            set_exit_status(1)
            raise FastExit()

        if rule_id and trigger:
            Interface.write_error("Only one of --rule-id or --trigger can be provided")
            set_exit_status(1)
            raise FastExit()

        Interface.add_journey_trigger(ctx, id, rule_id, trigger)

    @journey.command("remove-trigger", help="Remove a trigger from a journey")
    @click.option("--id", type=str, metavar="ID", help="Journey ID", required=True)
    @click.option("--trigger", type=str, metavar="TRIGGER", help="Trigger", required=True)
    @click.pass_context
    def journey_remove_trigger(ctx: click.Context, id: str, trigger: str) -> None:
        Interface.remove_journey_trigger(ctx, id, trigger)

    @journey.command("group", help="Group a journey")
    @click.option("--id", type=str, metavar="ID", help="Journey ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def journey_add_tag(
        ctx: click.Context,
        id: str,
        group: str,
    ) -> None:
        Interface.add_journey_group(
            ctx=ctx,
            journey_id=id,
            group=group,
        )

    @journey.command("untag", help="Untag from a journey")
    @click.option("--id", type=str, metavar="ID", help="Journey ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def journey_untag(ctx: click.Context, id: str, group: str) -> None:
        Interface.remove_journey_group(
            ctx=ctx,
            journey_id=id,
            group=group,
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
        ctx: click.Context, title: str, description: str, query: tuple[str], group: tuple[str]
    ) -> None:
        Interface.create_capability(ctx, title, description, list(query), list(group))

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
    def capability_list(ctx: click.Context, group: Optional[str]) -> None:
        Interface.list_capabilities(ctx, group)

    @capability.command("group", help="Group a capability")
    @click.option("--id", type=str, metavar="ID", help="Capability ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def capability_add_tag(
        ctx: click.Context,
        id: str,
        group: str,
    ) -> None:
        Interface.add_capability_tag(
            ctx=ctx,
            capability_id=id,
            group=group,
        )

    @capability.command("untag", help="Untag from a capability")
    @click.option("--id", type=str, metavar="ID", help="Capability ID", required=True)
    @tag_option(required=True)
    @click.pass_context
    def capability_untag(ctx: click.Context, id: str, group: str) -> None:
        Interface.remove_capability_tag(
            ctx=ctx,
            capability_id=id,
            group=group,
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
    @click.option("--rule-matcher", "-g", is_flag=True, help="Filter logs by [RuleMatcher]")
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
        rule_matcher: bool,
        tool_caller: bool,
        message_event_composer: bool,
        intersection_patterns: tuple[str],
        union_patterns: tuple[str],
    ) -> None:
        union_pattern_list = list(union_patterns)

        if rule_matcher:
            union_pattern_list.append("[RuleMatcher]")
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
