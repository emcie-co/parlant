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

"""The alpha engine is gone; nothing may import it back into existence."""

import subprocess
from pathlib import Path


def test_that_nothing_imports_the_removed_alpha_engine() -> None:
    repo = Path(__file__).parent.parent.parent.parent

    result = subprocess.run(
        [
            "grep",
            "-rln",
            r"parlant\.core\.engines\.alpha",
            "--include=*.py",
            str(repo / "src"),
            str(repo / "tests"),
        ],
        capture_output=True,
        text=True,
    )

    offenders = result.stdout.splitlines()

    assert not offenders, f"alpha engine imported: {offenders}"
