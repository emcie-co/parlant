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
from collections.abc import Iterator
import os
from pathlib import Path
import stat
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.core.common import DefaultBaseModel
from parlant.core.diagnostic_reports import DiagnosticReportService


class DiagnosticReportDTO(DefaultBaseModel):
    """One finalized engine-turn report available for administrator export."""

    trace_id: str
    session_id: str | None
    outcome: str
    created_utc: str
    finalized_utc: str
    size_bytes: int | None
    report_uri: str


def _stream_archive(
    path: Path,
    trace_id: str,
    *,
    delete_after_stream: bool,
) -> StreamingResponse:
    """Open a validated archive without following a substituted symlink."""

    def cleanup() -> None:
        if delete_after_stream:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            os.close(descriptor)
            cleanup()
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    except OSError as exc:
        cleanup()
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND) from exc

    def stream() -> Iterator[bytes]:
        try:
            with os.fdopen(descriptor, "rb") as report:
                while chunk := report.read(64 * 1024):
                    yield chunk
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass
            cleanup()

    return StreamingResponse(
        stream(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{trace_id}.zip"',
            "Content-Length": str(file_stat.st_size),
        },
        background=BackgroundTask(cleanup),
    )


def create_router(
    diagnostic_report_service: DiagnosticReportService,
    authorization_policy: AuthorizationPolicy,
) -> APIRouter:
    router = APIRouter()

    @router.get(
        "/debug/reports",
        operation_id="list_diagnostic_reports",
        response_model=list[DiagnosticReportDTO],
    )
    async def list_diagnostic_reports(
        request: Request,
        limit: Annotated[int | None, Query(ge=1)] = None,
        outcome: Literal["success", "error"] | None = None,
    ) -> list[DiagnosticReportDTO]:
        await authorization_policy.authorize(request, Operation.READ_DIAGNOSTIC_REPORT)

        return [
            DiagnosticReportDTO(
                trace_id=report.trace_id,
                session_id=report.session_id,
                outcome=report.outcome,
                created_utc=report.created_utc,
                finalized_utc=report.finalized_utc,
                size_bytes=report.size_bytes,
                report_uri=f"/debug/reports/{report.trace_id}",
            )
            for report in diagnostic_report_service.list_reports(limit, outcome)
        ]

    @router.get(
        "/debug/reports/{trace_id}",
        operation_id="read_diagnostic_report",
        response_class=StreamingResponse,
    )
    async def read_diagnostic_report(request: Request, trace_id: str) -> StreamingResponse:
        await authorization_policy.authorize(request, Operation.READ_DIAGNOSTIC_REPORT)

        prepared = await asyncio.to_thread(
            diagnostic_report_service.prepare_report,
            trace_id,
        )
        if prepared is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

        return _stream_archive(
            prepared.path,
            trace_id,
            delete_after_stream=prepared.delete_after_stream,
        )

    return router
