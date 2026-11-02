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

import asyncio
from typing import Any

import parlant.sdk as p
from parlant.core.agents import AgentId
from parlant.core.customers import CustomerStore
from parlant.core import metadata
from parlant.core.sessions import SessionId, SessionStore
from tests.sdk.utils import Context, SDKTest


class Test_that_compass_hooks_share_and_commit_the_turn_metadata_workspace(SDKTest):
    """Exercise workspace staging and SDK rebasing through a real Compass server."""

    internal_seen_after_refresh: object = None

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        async def on_acknowledged(
            context: p.EngineContext,
            payload: Any,
            exception: Exception | None,
        ) -> p.CompassHookResult:
            assert context.session_metadata is not None
            context.session_metadata.set_internal(
                ("compass", "sdk_test"),
                {"state": "staged"},
            )

            current = p.Session.current
            await current.metadata.set("hook_user_key", "preserved")

            # Async get performs a fresh store read. The staged internal value
            # must remain visible after that fresh snapshot is rebased.
            assert await current.metadata.get("hook_user_key") == "preserved"
            self.internal_seen_after_refresh = metadata.internal_value(
                {key: current.metadata[key] for key in current.metadata},
                ("compass", "sdk_test"),
            )
            return p.CompassHookResult.BAIL

        hooks.on_acknowledged.append(on_acknowledged)
        return hooks

    async def create_server(self, port: int) -> tuple[p.Server, Any]:
        server, get_container = await super().create_server(port)
        server.diagnostic_reports = False
        return server, get_container

    async def setup(self, server: p.Server) -> None:
        self.internal_seen_after_refresh = None
        self.agent = await server.create_agent(
            id="session-metadata-workspace-agent-1",
            name="Compass metadata workspace agent",
            instructions="Metadata workspace SDK test",
        )

    async def run(self, ctx: Context) -> None:
        store = ctx.container[SessionStore]
        session = await store.create_session(
            customer_id=CustomerStore.GUEST_ID,
            agent_id=AgentId(self.agent.id),
            metadata=metadata.add_internal_metadata({}, {}),
        )
        await ctx.client.sessions.create_event(
            session_id=str(session.id),
            kind="message",
            source="customer",
            message="Exercise metadata",
        )

        for _ in range(100):
            stored = await store.read_session(SessionId(session.id))
            if metadata.internal_value(stored.metadata, ("compass", "sdk_test")) is not None:
                break
            await asyncio.sleep(0.05)
        else:
            raise TimeoutError("Compass did not commit its metadata workspace")

        assert self.internal_seen_after_refresh == {"state": "staged"}
        assert stored.metadata["hook_user_key"] == "preserved"
        assert metadata.internal_value(stored.metadata, ("compass", "sdk_test")) == {
            "state": "staged"
        }

        sdk_session = await ctx.server.get_session(id=str(session.id))
        try:
            await sdk_session.metadata.set(metadata.PARLANT_METADATA_KEY, {})
        except ValueError as exc:
            assert "reserved for Parlant" in str(exc)
        else:
            raise AssertionError("The SDK accepted a write to reserved metadata")
