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

"""``parlant dashboard logs`` — the running project's log stream."""

from __future__ import annotations

import click

from parlant.bin.dashboard.auth import authenticate
from parlant.bin.dashboard.client import project_path, stream
from parlant.bin.dashboard.commands import dashboard
from parlant.bin.dashboard.commands._shared import require_project, runtime_for
from parlant.bin.dashboard.output import write_stream_line
from parlant.bin.help import examples


@dashboard.command(
    "logs",
    epilog=examples(
        "parlant dashboard logs",
        "parlant dashboard logs --project PROJECT_ID",
    ),
)
@click.pass_context
def logs(ctx: click.Context) -> None:
    """Follow the project's runtime logs.

    Runs until you stop it. A running project keeps producing logs, so there
    is no completion for this command to wait for.
    """
    runtime = runtime_for(ctx)
    project = require_project(runtime)

    try:
        for event, data in stream(
            runtime,
            project_path(project, "runtime-logs", "stream"),
            access_token=authenticate(runtime),
        ):
            if event in ("done", "end"):
                return
            write_stream_line(data)
    except KeyboardInterrupt:
        click.echo("")
