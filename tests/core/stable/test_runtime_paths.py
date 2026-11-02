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

from pathlib import Path

import pytest

from parlant.core.runtime_paths import (
    ERROR_REPORTS_DIR_ENV_VAR,
    DEFAULT_LOG_ROTATION_BACKUP_COUNT,
    DEFAULT_LOG_ROTATION_MAX_BYTES,
    LOG_ROTATION_BACKUP_COUNT_ENV_VAR,
    LOG_ROTATION_MAX_BYTES_ENV_VAR,
    LOGS_DIR_ENV_VAR,
    PARLANT_HOME_ENV_VAR,
    TRACE_REPORTS_DIR_ENV_VAR,
    resolve_log_rotation_config,
    resolve_runtime_paths,
)


def test_that_telemetry_paths_default_under_parlant_home() -> None:
    paths = resolve_runtime_paths("parlant-data", {})

    assert paths.home_dir == Path("parlant-data")
    assert paths.error_reports_dir == Path("parlant-data/diagnostic-reports/errors")
    assert paths.trace_reports_dir == Path("parlant-data/diagnostic-reports/traces")
    assert paths.logs_dir == Path("parlant-data/logs")
    assert paths.log_file == Path("parlant-data/logs/parlant.log")


def test_that_each_telemetry_directory_can_override_the_home_default() -> None:
    paths = resolve_runtime_paths(
        "unused",
        {
            PARLANT_HOME_ENV_VAR: "/srv/parlant",
            ERROR_REPORTS_DIR_ENV_VAR: "/diagnostics/errors",
            TRACE_REPORTS_DIR_ENV_VAR: "/diagnostics/traces",
            LOGS_DIR_ENV_VAR: "/var/log/parlant",
        },
    )

    assert paths.home_dir == Path("/srv/parlant")
    assert paths.error_reports_dir == Path("/diagnostics/errors")
    assert paths.trace_reports_dir == Path("/diagnostics/traces")
    assert paths.logs_dir == Path("/var/log/parlant")
    assert paths.log_file == Path("/var/log/parlant/parlant.log")


def test_that_relative_directory_overrides_remain_relative_to_the_working_directory() -> None:
    paths = resolve_runtime_paths(
        "unused",
        {LOGS_DIR_ENV_VAR: "deployment-logs"},
    )

    assert paths.logs_dir == Path("deployment-logs")


@pytest.mark.parametrize(
    "name",
    [
        PARLANT_HOME_ENV_VAR,
        ERROR_REPORTS_DIR_ENV_VAR,
        TRACE_REPORTS_DIR_ENV_VAR,
        LOGS_DIR_ENV_VAR,
    ],
)
def test_that_empty_path_overrides_are_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        resolve_runtime_paths("parlant-data", {name: ""})


def test_that_log_rotation_has_bounded_defaults_and_honors_overrides() -> None:
    defaults = resolve_log_rotation_config({})
    configured = resolve_log_rotation_config(
        {
            LOG_ROTATION_MAX_BYTES_ENV_VAR: "2048",
            LOG_ROTATION_BACKUP_COUNT_ENV_VAR: "3",
        }
    )
    disabled = resolve_log_rotation_config(
        {
            LOG_ROTATION_MAX_BYTES_ENV_VAR: "0",
            LOG_ROTATION_BACKUP_COUNT_ENV_VAR: "0",
        }
    )

    assert defaults.max_bytes == DEFAULT_LOG_ROTATION_MAX_BYTES == 10 * 1024 * 1024
    assert defaults.backup_count == DEFAULT_LOG_ROTATION_BACKUP_COUNT == 5
    assert configured.max_bytes == 2048
    assert configured.backup_count == 3
    assert disabled.max_bytes == 0
    assert disabled.backup_count == 0


@pytest.mark.parametrize(
    ("environ", "expected_name"),
    [
        ({LOG_ROTATION_MAX_BYTES_ENV_VAR: "invalid"}, LOG_ROTATION_MAX_BYTES_ENV_VAR),
        ({LOG_ROTATION_MAX_BYTES_ENV_VAR: "-1"}, LOG_ROTATION_MAX_BYTES_ENV_VAR),
        ({LOG_ROTATION_BACKUP_COUNT_ENV_VAR: "-1"}, LOG_ROTATION_BACKUP_COUNT_ENV_VAR),
        (
            {
                LOG_ROTATION_MAX_BYTES_ENV_VAR: "1024",
                LOG_ROTATION_BACKUP_COUNT_ENV_VAR: "0",
            },
            LOG_ROTATION_BACKUP_COUNT_ENV_VAR,
        ),
    ],
)
def test_that_invalid_log_rotation_configuration_is_rejected(
    environ: dict[str, str],
    expected_name: str,
) -> None:
    with pytest.raises(ValueError, match=expected_name):
        resolve_log_rotation_config(environ)
