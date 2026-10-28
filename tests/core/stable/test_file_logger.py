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

import os
from pathlib import Path
from uuid import uuid4

import pytest

from parlant.core.loggers import FileLogger, LogLevel
from parlant.core.tracer import LocalTracer


def _close_handlers(logger: FileLogger) -> None:
    for handler in list(logger.raw_logger.handlers):
        handler.flush()
        handler.close()
        logger.raw_logger.removeHandler(handler)


def test_that_file_logger_rotates_by_size_and_prunes_old_backups(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "parlant.log"
    logger = FileLogger(
        log_file,
        LocalTracer(),
        LogLevel.INFO,
        logger_id=f"rotation-test-{uuid4()}",
        max_bytes=256,
        backup_count=2,
    )
    try:
        for index in range(20):
            logger.info(f"rotation-sentinel-{index}-" + ("x" * 120))
    finally:
        _close_handlers(logger)

    files = sorted(log_file.parent.glob("parlant.log*"))
    assert [path.name for path in files] == [
        "parlant.log",
        "parlant.log.1",
        "parlant.log.2",
    ]
    assert "rotation-sentinel-19" in log_file.read_text()
    if os.name == "posix":
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in files)


def test_that_zero_max_bytes_disables_log_rotation(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "parlant.log"
    logger = FileLogger(
        log_file,
        LocalTracer(),
        logger_id=f"no-rotation-test-{uuid4()}",
        max_bytes=0,
        backup_count=0,
    )
    try:
        for _ in range(20):
            logger.info("x" * 120)
    finally:
        _close_handlers(logger)

    assert [path.name for path in log_file.parent.glob("parlant.log*")] == ["parlant.log"]


def test_that_file_logger_creates_owner_only_log_storage(tmp_path: Path) -> None:
    log_file = tmp_path / "new-logs" / "parlant.log"
    logger = FileLogger(
        log_file,
        LocalTracer(),
        logger_id=f"permissions-test-{uuid4()}",
    )
    _close_handlers(logger)

    if os.name == "posix":
        assert log_file.parent.stat().st_mode & 0o777 == 0o700
        assert log_file.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize(
    ("max_bytes", "backup_count"),
    [(-1, 1), (1, -1), (1, 0)],
)
def test_that_file_logger_rejects_invalid_rotation_settings(
    tmp_path: Path,
    max_bytes: int,
    backup_count: int,
) -> None:
    log_file = tmp_path / "parlant.log"
    with pytest.raises(ValueError):
        FileLogger(
            log_file,
            LocalTracer(),
            logger_id=f"invalid-rotation-test-{uuid4()}",
            max_bytes=max_bytes,
            backup_count=backup_count,
        )

    assert not log_file.exists()
