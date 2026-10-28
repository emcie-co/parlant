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
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Callable, ClassVar, Mapping, Sequence, cast
import zipfile

import httpx
from parlant.core.container import Singleton

import parlant.sdk as p
from parlant.core.diagnostic_reports import (
    DiagnosticReportRepository,
    DiagnosticTraceHistory,
)
from parlant.core.emissions import EventEmitter
from parlant.core.engines.compass.engine import CompassEngine
from parlant.core.engines.compass.context import EngineContext
from parlant.core.engines.types import Context as EngineInfo
from parlant.core.engines.types import UtteranceRequest
from parlant.core import metadata

from tests.sdk.utils import Context, SDKTest


PROCESS_SENTINEL = "sdk-process-diagnostic-report-sentinel"
UTTER_SENTINEL = "sdk-utter-diagnostic-report-sentinel"


class _DeterministicallyFailingCompassEngine(CompassEngine):
    """A fully wired CompassEngine that fails deterministically inside the
    engine's guarded regions — the ones diagnostic reporting captures."""

    _fail_utter: ClassVar[bool] = False

    async def initialize(
        self,
        context: EngineInfo,
        event_emitter: EventEmitter,
    ) -> None:
        # Skip warm-ups: these tests never reach response generation.
        pass

    async def _call_rule_handlers(self, context: EngineContext) -> None:
        # The first engine-owned step inside process()'s guarded region.
        if self._fail_utter:
            await super()._call_rule_handlers(context)
            return
        self._logger.info("SDK process reached the deterministic failure")
        raise RuntimeError(PROCESS_SENTINEL)

    async def _do_utter(
        self,
        engine_context: EngineContext,
        requests: Sequence[UtteranceRequest],
    ) -> bool:
        if not self._fail_utter:
            raise AssertionError("This SDK test engine was not configured for utter")
        self._logger.info("SDK utter reached the deterministic failure")
        raise RuntimeError(UTTER_SENTINEL)


class _UtterFailingCompassEngine(_DeterministicallyFailingCompassEngine):
    _fail_utter: ClassVar[bool] = True


class _DeterministicallySuccessfulCompassEngine(_DeterministicallyFailingCompassEngine):
    async def process(
        self,
        context: EngineInfo,
        event_emitter: EventEmitter,
    ) -> bool:
        with self._tracer.span(
            "process",
            {"session_id": context.session_id, "agent_id": context.agent_id},
        ):
            self._logger.info("SDK process reached deterministic success")
            await event_emitter.emit_status_event(
                trace_id=self._tracer.trace_id,
                data={"status": "ready", "data": {"stage": "completed"}},
            )
        return True


class _DeterministicallySuccessfulUtterCompassEngine(_DeterministicallyFailingCompassEngine):
    async def utter(
        self,
        context: EngineInfo,
        event_emitter: EventEmitter,
        requests: Sequence[UtteranceRequest],
    ) -> bool:
        with self._tracer.span(
            "utter",
            {"session_id": context.session_id, "agent_id": context.agent_id},
        ):
            self._logger.info("SDK utter reached deterministic success")
            await event_emitter.emit_message_event(
                trace_id=self._tracer.trace_id,
                data="Deterministic successful utter",
            )
        return True


class _DiagnosticReportSDKTest(SDKTest):
    disable_diagnostic_reports = False
    fail_utter = False
    successful_process = False
    successful_utter = False
    trace_history_size = 0
    error_history_size = 8

    async def test_run(self) -> None:
        self._temporary_reports = TemporaryDirectory()
        self.report_dir = Path(self._temporary_reports.name)
        try:
            await super().test_run()
        finally:
            self._temporary_reports.cleanup()

    async def create_server(
        self,
        port: int,
    ) -> tuple[p.Server, Callable[[], p.Container]]:
        server, get_container = await super().create_server(port)
        server.diagnostic_reports = not self.disable_diagnostic_reports
        return server, get_container

    async def configure_container(self, container: p.Container) -> p.Container:
        container[DiagnosticReportRepository] = DiagnosticReportRepository(
            error_report_dir=self.report_dir,
            trace_report_dir=self.report_dir / "traces",
            error_history_size=self.error_history_size,
        )
        container[DiagnosticTraceHistory] = DiagnosticTraceHistory(self.trace_history_size)
        engine_type: type[_DeterministicallyFailingCompassEngine]
        if self.successful_process:
            engine_type = _DeterministicallySuccessfulCompassEngine
        elif self.successful_utter:
            engine_type = _DeterministicallySuccessfulUtterCompassEngine
        elif self.fail_utter:
            engine_type = _UtterFailingCompassEngine
        else:
            engine_type = _DeterministicallyFailingCompassEngine
        container[engine_type] = Singleton(engine_type)
        container[CompassEngine] = lambda c: c[engine_type]
        return container

    async def _wait_for_error(
        self,
        ctx: Context,
        session_id: str,
        trace_id: str,
    ) -> Any:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            events = await ctx.client.sessions.list_events(
                session_id=session_id,
                trace_id=trace_id,
                kinds="status",
                wait_for_data=0,
            )
            if error := next(
                (
                    event
                    for event in events
                    if isinstance(event.data, dict) and event.data.get("status") == "error"
                ),
                None,
            ):
                return error
            await asyncio.sleep(0.05)

        raise TimeoutError("The engine did not emit an error status")

    async def _download_completed_report(
        self,
        ctx: Context,
        report_uri: str,
    ) -> tuple[httpx.Response, Mapping[str, bytes]]:
        deadline = time.monotonic() + 10
        async with httpx.AsyncClient(base_url=f"http://localhost:{ctx.server.port}") as http:
            while time.monotonic() < deadline:
                response = await http.get(report_uri)
                if response.status_code == 200:
                    with zipfile.ZipFile(BytesIO(response.content)) as archive:
                        members = {name: archive.read(name) for name in archive.namelist()}
                    if json.loads(members["manifest.json"])["complete"]:
                        return response, members
                await asyncio.sleep(0.05)

        raise TimeoutError("The diagnostic report was not finalized")

    async def _download_successful_report(
        self,
        ctx: Context,
        trace_id: str,
    ) -> tuple[httpx.Response, Mapping[str, bytes]]:
        deadline = time.monotonic() + 10
        async with httpx.AsyncClient(base_url=f"http://localhost:{ctx.server.port}") as http:
            while time.monotonic() < deadline:
                listing = await http.get("/debug/reports", params={"outcome": "success"})
                assert listing.status_code == 200
                if any(item["trace_id"] == trace_id for item in listing.json()):
                    response = await http.get(f"/debug/reports/{trace_id}")
                    assert response.status_code == 200
                    with zipfile.ZipFile(BytesIO(response.content)) as archive:
                        return response, {name: archive.read(name) for name in archive.namelist()}
                await asyncio.sleep(0.05)

        raise TimeoutError("The successful diagnostic report was not finalized")


class Test_that_an_sdk_server_exports_a_successful_background_process_trace(
    _DiagnosticReportSDKTest
):
    successful_process = True
    trace_history_size = 2

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="diagnostic-reports-agent-1",
            name="Successful Trace Export Agent",
            instructions="This agent succeeds without using a model.",
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )
        source_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Trigger deterministic process success",
        )

        response, members = await self._download_successful_report(
            ctx,
            source_event.trace_id,
        )
        manifest = json.loads(members["manifest.json"])
        trace = json.loads(members["trace.json"])

        assert response.headers["content-disposition"] == (
            f'attachment; filename="{source_event.trace_id}.zip"'
        )
        assert "traceback.txt" not in members
        assert manifest["trace_id"] == source_event.trace_id
        assert manifest["session_id"] == session.id
        assert manifest["outcome"] == "success"
        assert manifest["complete"] is True
        assert "SDK process reached deterministic success" in members["logs.jsonl"].decode()
        assert "session.json" in members
        assert "events.jsonl" in members
        assert {span["name"] for span in trace["spans"]} >= {"http.request", "process"}
        assert list((self.report_dir / "traces").iterdir()) == []


class Test_that_an_sdk_server_exports_a_successful_inline_utter_trace(_DiagnosticReportSDKTest):
    successful_utter = True
    trace_history_size = 2

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="diagnostic-reports-agent-2",
            name="Successful Utter Trace Export Agent",
            instructions="This agent utters without using a model.",
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )

        async with httpx.AsyncClient(base_url=f"http://localhost:{ctx.server.port}") as http:
            utter_response = await http.post(
                f"/sessions/{session.id}/events",
                json={
                    "kind": "message",
                    "source": "ai_agent",
                    "rules": [
                        {
                            "action": "Say anything",
                            "rationale": "unspecified",
                        }
                    ],
                },
            )

        assert utter_response.status_code == 201
        event = utter_response.json()
        assert event["kind"] == "message"
        trace_id = event["trace_id"]

        _response, members = await self._download_successful_report(ctx, trace_id)
        manifest = json.loads(members["manifest.json"])
        trace = json.loads(members["trace.json"])

        assert "traceback.txt" not in members
        assert manifest["outcome"] == "success"
        assert manifest["session_id"] == session.id
        assert "SDK utter reached deterministic success" in members["logs.jsonl"].decode()
        assert "session.json" in members
        assert "events.jsonl" in members
        span_names = [span["name"] for span in trace["spans"]]
        assert "sessions.utter" in span_names
        # The stub engine emits its own bare "utter" span inside the
        # session module's sessions.utter wrapper.
        assert "utter" in span_names
        assert list((self.report_dir / "traces").iterdir()) == []


class Test_that_an_sdk_server_captures_the_full_background_process_trace(_DiagnosticReportSDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="diagnostic-reports-agent-3",
            name="Diagnostic Report Process Agent",
            instructions="This agent fails before using a model.",
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )
        source_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Trigger deterministic process failure",
        )
        error_event = await self._wait_for_error(
            ctx,
            session_id=session.id,
            trace_id=source_event.trace_id,
        )

        report_uri = metadata.internal_value(error_event.metadata, ("diagnostic_report_uri",))
        assert report_uri == f"/debug/reports/{source_event.trace_id}"
        assert error_event.trace_id == source_event.trace_id

        response, members = await self._download_completed_report(
            ctx,
            report_uri,
        )
        manifest = json.loads(members["manifest.json"])
        logs = [json.loads(line) for line in members["logs.jsonl"].decode().splitlines()]
        trace = json.loads(members["trace.json"])

        assert response.headers["content-disposition"] == (
            f'attachment; filename="{source_event.trace_id}.zip"'
        )
        assert PROCESS_SENTINEL in members["traceback.txt"].decode()
        assert manifest["trace_id"] == source_event.trace_id
        assert manifest["complete"] is True
        assert any("Dispatching processing" in record["message"] for record in logs)
        assert any(PROCESS_SENTINEL in record["message"] for record in logs)
        span_names = [span["name"] for span in trace["spans"]]
        assert "http.request" in span_names
        assert "compass.engine.process" in span_names
        assert list(self.report_dir.glob("*.zip")) == [
            self.report_dir / f"{source_event.trace_id}.zip"
        ]


class Test_that_an_sdk_server_returns_and_reports_an_inline_utter_failure(_DiagnosticReportSDKTest):
    fail_utter = True

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="diagnostic-reports-agent-4",
            name="Diagnostic Report Utter Agent",
            instructions="This agent fails in its deterministic message generator.",
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )

        async with httpx.AsyncClient(base_url=f"http://localhost:{ctx.server.port}") as http:
            response = await http.post(
                f"/sessions/{session.id}/events",
                json={
                    "kind": "message",
                    "source": "ai_agent",
                    "rules": [
                        {
                            "action": "Say anything",
                            "rationale": "unspecified",
                        }
                    ],
                },
            )

        assert response.status_code == 201
        error_event = response.json()
        assert error_event["kind"] == "status"
        assert error_event["data"]["status"] == "error"
        report_uri = metadata.internal_value(error_event["metadata"], ("diagnostic_report_uri",))
        trace_id = error_event["trace_id"]
        assert report_uri == f"/debug/reports/{trace_id}"

        _response, members = await self._download_completed_report(ctx, report_uri)
        trace = json.loads(members["trace.json"])
        logs = members["logs.jsonl"].decode()

        assert UTTER_SENTINEL in members["traceback.txt"].decode()
        assert "SDK utter reached the deterministic failure" in logs
        span_names = [span["name"] for span in trace["spans"]]
        assert "sessions.utter" in span_names
        assert "compass.engine.utter" in span_names
        assert "http.request" in span_names


class Test_that_disabling_diagnostic_reports_omits_reports_and_metadata(_DiagnosticReportSDKTest):
    disable_diagnostic_reports = True

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="diagnostic-reports-agent-5",
            name="Disabled Diagnostic Report Agent",
            instructions="This agent fails before using a model.",
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )
        source_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Trigger deterministic process failure",
        )
        error_event = await self._wait_for_error(
            ctx,
            session_id=session.id,
            trace_id=source_event.trace_id,
        )

        assert metadata.internal_value(error_event.metadata, ("diagnostic_report_uri",)) is None
        assert not list(self.report_dir.glob("*.zip"))

        async with httpx.AsyncClient(base_url=f"http://localhost:{ctx.server.port}") as http:
            response = await http.get(f"/debug/reports/{source_event.trace_id}")

        assert response.status_code == 404


class Test_that_zero_histories_disable_reports_and_metadata(_DiagnosticReportSDKTest):
    error_history_size = 0
    trace_history_size = 0

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="diagnostic-reports-agent-6",
            name="Zero Diagnostic History Agent",
            instructions="This agent errors before using a model.",
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )
        source_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Trigger deterministic process error",
        )
        error_event = await self._wait_for_error(
            ctx,
            session_id=session.id,
            trace_id=source_event.trace_id,
        )

        assert metadata.internal_value(error_event.metadata, ("diagnostic_report_uri",)) is None
        assert not list(self.report_dir.rglob("*.zip"))

        async with httpx.AsyncClient(base_url=f"http://localhost:{ctx.server.port}") as http:
            response = await http.get(f"/debug/reports/{source_event.trace_id}")

        assert response.status_code == 404


class Test_that_a_completed_report_exports_the_sessions_events_and_metadata(
    _DiagnosticReportSDKTest
):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="diagnostic-reports-agent-7",
            name="Diagnostic Report Session Export Agent",
            instructions="This agent fails before using a model.",
        )

    async def run(self, ctx: Context) -> None:
        session = await ctx.client.sessions.create(
            agent_id=self.agent.id,
            allow_greeting=False,
        )
        source_event = await ctx.client.sessions.create_event(
            session_id=session.id,
            kind="message",
            source="customer",
            message="Trigger deterministic process failure",
        )
        error_event = await self._wait_for_error(
            ctx,
            session_id=session.id,
            trace_id=source_event.trace_id,
        )

        _, members = await self._download_completed_report(
            ctx,
            cast(
                str,
                metadata.internal_value(error_event.metadata, ("diagnostic_report_uri",)),
            ),
        )

        manifest = json.loads(members["manifest.json"])
        assert "session.json" in members
        assert "events.jsonl" in members
        assert "session.json" in manifest["files"]
        assert "events.jsonl" in manifest["files"]

        session_doc = json.loads(members["session.json"])
        assert session_doc["id"] == session.id
        assert session_doc["agent_id"] == self.agent.id
        assert "metadata" in session_doc

        events = [json.loads(line) for line in members["events.jsonl"].decode().splitlines()]
        assert any(
            event["kind"] == "message"
            and "Trigger deterministic process failure" in json.dumps(event["data"])
            for event in events
        )
        assert any(
            event["kind"] == "status"
            and isinstance(event["data"], dict)
            and event["data"].get("status") == "error"
            for event in events
        )
