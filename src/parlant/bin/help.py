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

"""Worked examples for `--help`.

A command's description says what it does; an example says what to type. The
second is what someone reaching for `--help` under time pressure actually
wants, and it is the only part a coding agent can copy.
"""

from __future__ import annotations


def examples(*invocations: str) -> str:
    r"""An epilog Click prints verbatim, one command line per entry.

    Click rewraps help text to the terminal width, which would run command
    lines together. The ``\b`` line disables that for the paragraph after it,
    so each example stays on its own line.
    """
    listed = "\n".join(f"  {invocation}" for invocation in invocations)

    return f"Examples:\n\n\b\n{listed}"
