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
import threading
from typing import AsyncIterator
import zipfile

import httpx
from parlant.core.container import Container
import pytest

from parlant.api.app import create_api_app
from parlant.core.diagnostic_reports import (
    DiagnosticReportRepository,
    DiagnosticReportService,
    DiagnosticTraceHistory,
    DiagnosticTraceSnapshot,
    TraceError,
)


@pytest.fixture
async def diagnostic_report_client(
    container: Container,
    tmp_path: Path,
) -> AsyncIterator[tuple[httpx.AsyncClient, DiagnosticReportRepository, DiagnosticTraceHistory]]:
    configured_container = container.clone()
    repository = DiagnosticReportRepository(
        error_report_dir=tmp_path / "errors",
        trace_report_dir=tmp_path / "traces",
        error_history_size=8,
    )
    history = DiagnosticTraceHistory(8)
    configured_container[DiagnosticReportService] = DiagnosticReportService(
        repository=repository,
        history=history,
    )
    app = await create_api_app(configured_container)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client, repository, history


def _error_snapshot(trace_id: str) -> DiagnosticTraceSnapshot:
    return DiagnosticTraceSnapshot(
        trace_id=trace_id,
        created_utc="2026-01-01T00:00:00+00:00",
        logs=(),
        spans=(),
        error=TraceError(
            traceback="Traceback\nRuntimeError: download sentinel\n",
            exception_type="RuntimeError",
            exception_message="download sentinel",
            captured_utc="2026-01-01T00:00:01+00:00",
            session_id="session-1",
        ),
    )


def _write_error_report(repository: DiagnosticReportRepository, trace_id: str) -> Path:
    return repository.write_error(
        _error_snapshot(trace_id),
        complete=True,
        finalized_utc="2026-01-01T00:00:02+00:00",
    )


def _retain_successful_trace(
    history: DiagnosticTraceHistory,
    trace_id: str,
    session_id: str,
) -> None:
    history.add(
        DiagnosticTraceSnapshot(
            trace_id=trace_id,
            created_utc="2026-01-01T00:00:00+00:00",
            logs=(),
            spans=(),
            engine_turn=True,
            session_id=session_id,
        ),
        finalized_utc="2026-01-01T00:00:01+00:00",
    )


async def test_that_report_endpoint_downloads_the_requested_archive(
    diagnostic_report_client: tuple[
        httpx.AsyncClient, DiagnosticReportRepository, DiagnosticTraceHistory
    ],
) -> None:
    client, repository, _history = diagnostic_report_client
    expected = _write_error_report(repository, "download-trace")

    response = await client.get("/debug/reports/download-trace")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == ('attachment; filename="download-trace.zip"')
    assert response.content == expected.read_bytes()

    with zipfile.ZipFile(expected) as archive:
        assert "RuntimeError: download sentinel" in archive.read("traceback.txt").decode()


async def test_that_an_initial_error_is_downloadable_but_not_listed_until_finalized(
    diagnostic_report_client: tuple[
        httpx.AsyncClient, DiagnosticReportRepository, DiagnosticTraceHistory
    ],
) -> None:
    client, repository, _history = diagnostic_report_client
    snapshot = _error_snapshot("pending-error")
    repository.write_error(snapshot, complete=False)

    initial_download = await client.get("/debug/reports/pending-error")
    initial_listing = await client.get("/debug/reports", params={"outcome": "error"})

    assert initial_download.status_code == 200
    with zipfile.ZipFile(BytesIO(initial_download.content)) as archive:
        assert json.loads(archive.read("manifest.json"))["complete"] is False
    assert initial_listing.status_code == 200
    assert initial_listing.json() == []

    repository.write_error(
        snapshot,
        complete=True,
        finalized_utc="2026-01-01T00:00:02+00:00",
    )
    final_listing = await client.get("/debug/reports", params={"outcome": "error"})

    assert final_listing.status_code == 200
    assert [report["trace_id"] for report in final_listing.json()] == ["pending-error"]


async def test_that_report_endpoint_rejects_missing_and_malformed_ids(
    diagnostic_report_client: tuple[
        httpx.AsyncClient, DiagnosticReportRepository, DiagnosticTraceHistory
    ],
) -> None:
    client, _repository, _history = diagnostic_report_client

    missing = await client.get("/debug/reports/missing")
    traversal = await client.get("/debug/reports/%2E%2E%2Foutside")
    invalid = await client.get("/debug/reports/not.valid")

    assert missing.status_code == 404
    assert traversal.status_code == 404
    assert invalid.status_code == 404


async def test_that_report_endpoint_rejects_a_symlink(
    diagnostic_report_client: tuple[
        httpx.AsyncClient, DiagnosticReportRepository, DiagnosticTraceHistory
    ],
    tmp_path: Path,
) -> None:
    client, repository, _history = diagnostic_report_client
    repository.error_report_dir.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"outside")

    try:
        (repository.error_report_dir / "linked.zip").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not supported on this platform")

    response = await client.get("/debug/reports/linked")

    assert response.status_code == 404


async def test_that_disabled_diagnostic_reports_return_not_found(
    async_client: httpx.AsyncClient,
) -> None:
    response = await async_client.get("/debug/reports/anything")

    assert response.status_code == 404


async def test_that_reports_are_listed_newest_first_with_report_uris(
    diagnostic_report_client: tuple[
        httpx.AsyncClient, DiagnosticReportRepository, DiagnosticTraceHistory
    ],
) -> None:
    client, repository, history = diagnostic_report_client
    _retain_successful_trace(history, "trace-first", "session-1")
    second = _write_error_report(repository, "trace-second")

    response = await client.get("/debug/reports")

    assert response.status_code == 200
    reports = response.json()
    assert [item["trace_id"] for item in reports] == ["trace-second", "trace-first"]
    assert reports[0] == {
        "trace_id": "trace-second",
        "session_id": "session-1",
        "outcome": "error",
        "created_utc": "2026-01-01T00:00:00+00:00",
        "finalized_utc": reports[0]["finalized_utc"],
        "size_bytes": second.stat().st_size,
        "report_uri": "/debug/reports/trace-second",
    }

    limited = await client.get("/debug/reports", params={"limit": 1})
    assert limited.status_code == 200
    assert [item["trace_id"] for item in limited.json()] == ["trace-second"]

    filtered = await client.get("/debug/reports", params={"outcome": "success"})
    assert filtered.status_code == 200
    assert [item["trace_id"] for item in filtered.json()] == ["trace-first"]
    assert filtered.json()[0]["size_bytes"] is None


async def test_that_successful_report_can_be_downloaded(
    diagnostic_report_client: tuple[
        httpx.AsyncClient, DiagnosticReportRepository, DiagnosticTraceHistory
    ],
) -> None:
    client, repository, history = diagnostic_report_client
    _retain_successful_trace(history, "successful-trace", "session-1")
    assert not repository.trace_report_dir.exists()

    response = await client.get("/debug/reports/successful-trace")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["content-disposition"] == (
        'attachment; filename="successful-trace.zip"'
    )
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert "traceback.txt" not in archive.namelist()
    assert list(repository.trace_report_dir.iterdir()) == []


async def test_that_concurrent_successful_exports_use_disposable_archives(
    diagnostic_report_client: tuple[
        httpx.AsyncClient, DiagnosticReportRepository, DiagnosticTraceHistory
    ],
) -> None:
    client, repository, history = diagnostic_report_client
    _retain_successful_trace(history, "concurrent-trace", "session-1")

    responses = await asyncio.gather(
        client.get("/debug/reports/concurrent-trace"),
        client.get("/debug/reports/concurrent-trace"),
    )

    assert [response.status_code for response in responses] == [200, 200]
    for response in responses:
        with zipfile.ZipFile(BytesIO(response.content)) as archive:
            assert "traceback.txt" not in archive.namelist()
    assert list(repository.trace_report_dir.iterdir()) == []


async def test_that_slow_materialization_does_not_block_other_requests(
    diagnostic_report_client: tuple[
        httpx.AsyncClient, DiagnosticReportRepository, DiagnosticTraceHistory
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, repository, history = diagnostic_report_client
    _retain_successful_trace(history, "slow-export", "session-1")
    materialization_started = threading.Event()
    continue_materialization = threading.Event()
    original_materialize = repository.materialize

    def slow_materialize(
        snapshot: DiagnosticTraceSnapshot,
        finalized_utc: str,
    ) -> Path:
        materialization_started.set()
        if not continue_materialization.wait(timeout=5):
            raise TimeoutError("test did not release diagnostic materialization")
        return original_materialize(snapshot, finalized_utc)

    monkeypatch.setattr(repository, "materialize", slow_materialize)
    download_task = asyncio.create_task(client.get("/debug/reports/slow-export"))

    async def wait_for_materialization() -> None:
        while not materialization_started.is_set():
            await asyncio.sleep(0.001)

    try:
        await asyncio.wait_for(wait_for_materialization(), timeout=1)
        listing = await asyncio.wait_for(client.get("/debug/reports"), timeout=1)
    finally:
        continue_materialization.set()
        download = await download_task

    assert listing.status_code == 200
    assert download.status_code == 200
    assert list(repository.trace_report_dir.iterdir()) == []


async def test_that_report_listing_rejects_invalid_requests(
    diagnostic_report_client: tuple[
        httpx.AsyncClient, DiagnosticReportRepository, DiagnosticTraceHistory
    ],
) -> None:
    client, _repository, _history = diagnostic_report_client

    invalid_limit = await client.get("/debug/reports", params={"limit": 0})
    invalid_outcome = await client.get("/debug/reports", params={"outcome": "pending"})

    assert invalid_limit.status_code == 422
    assert invalid_outcome.status_code == 422
