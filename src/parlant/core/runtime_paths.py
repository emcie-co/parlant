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

"""Environment-backed filesystem configuration for local telemetry artifacts."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


PARLANT_HOME_ENV_VAR = "PARLANT_HOME"
ERROR_REPORTS_DIR_ENV_VAR = "PARLANT_ERROR_REPORTS_DIR"
TRACE_REPORTS_DIR_ENV_VAR = "PARLANT_TRACE_REPORTS_DIR"
LOGS_DIR_ENV_VAR = "PARLANT_LOGS_DIR"
LOG_ROTATION_MAX_BYTES_ENV_VAR = "PARLANT_LOG_ROTATION_MAX_BYTES"
LOG_ROTATION_BACKUP_COUNT_ENV_VAR = "PARLANT_LOG_ROTATION_BACKUP_COUNT"

DEFAULT_LOG_ROTATION_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_LOG_ROTATION_BACKUP_COUNT = 5


@dataclass(frozen=True)
class RuntimePaths:
    """Resolved local paths for Parlant state and telemetry artifacts."""

    home_dir: Path
    error_reports_dir: Path
    trace_reports_dir: Path
    logs_dir: Path

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "parlant.log"


@dataclass(frozen=True)
class LogRotationConfig:
    """Size-based rotation settings for Parlant's local log file."""

    max_bytes: int
    backup_count: int


def resolve_runtime_paths(
    default_home_dir: str | Path,
    environ: Mapping[str, str] | None = None,
) -> RuntimePaths:
    """Resolve individual telemetry directories with home-based fallbacks."""

    values = os.environ if environ is None else environ
    home_dir = _resolve_path(values, PARLANT_HOME_ENV_VAR, Path(default_home_dir))
    return RuntimePaths(
        home_dir=home_dir,
        error_reports_dir=_resolve_path(
            values,
            ERROR_REPORTS_DIR_ENV_VAR,
            home_dir / "diagnostic-reports" / "errors",
        ),
        trace_reports_dir=_resolve_path(
            values,
            TRACE_REPORTS_DIR_ENV_VAR,
            home_dir / "diagnostic-reports" / "traces",
        ),
        logs_dir=_resolve_path(
            values,
            LOGS_DIR_ENV_VAR,
            home_dir / "logs",
        ),
    )


def resolve_log_rotation_config(
    environ: Mapping[str, str] | None = None,
) -> LogRotationConfig:
    """Resolve and validate size-based local log rotation settings."""

    values = os.environ if environ is None else environ
    max_bytes = _resolve_nonnegative_int(
        values,
        LOG_ROTATION_MAX_BYTES_ENV_VAR,
        DEFAULT_LOG_ROTATION_MAX_BYTES,
    )
    backup_count = _resolve_nonnegative_int(
        values,
        LOG_ROTATION_BACKUP_COUNT_ENV_VAR,
        DEFAULT_LOG_ROTATION_BACKUP_COUNT,
    )

    if max_bytes > 0 and backup_count < 1:
        raise ValueError(
            f"{LOG_ROTATION_BACKUP_COUNT_ENV_VAR} must be at least 1 when "
            f"{LOG_ROTATION_MAX_BYTES_ENV_VAR} is greater than 0"
        )

    return LogRotationConfig(max_bytes=max_bytes, backup_count=backup_count)


def _resolve_path(
    environ: Mapping[str, str],
    name: str,
    default: Path,
) -> Path:
    value = environ.get(name)
    if value is None:
        return default
    if not value:
        raise ValueError(f"{name} must not be empty")
    return Path(value).expanduser()


def _resolve_nonnegative_int(
    environ: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    raw_value = environ.get(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc

    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")

    return value
