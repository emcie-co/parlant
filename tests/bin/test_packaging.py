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

from importlib.metadata import distributions
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# The widget source is ~600KB; widget/node_modules is ~127MB. Bounding the
# widget's own contribution turns "we forgot to exclude the toolchain" into a
# failing test rather than a surprise on PyPI. It deliberately measures the
# widget subtree rather than the whole wheel, so unrelated payloads (the chat
# UI is tens of megabytes) can't mask or trip this guard.
MAX_WIDGET_MEGABYTES = 5


def test_that_installed_dependencies_do_not_overwrite_griffe_module_files() -> None:
    owners: dict[str, set[str]] = {}
    for distribution in distributions():
        for path in distribution.files or ():
            if str(path).startswith("griffe/") and str(path).endswith(".py"):
                owners.setdefault(str(path), set()).add(distribution.metadata["Name"])

    assert owners, "FastMCP's Griffe modules must be installed"
    overlaps = {path: names for path, names in owners.items() if len(names) > 1}
    assert not overlaps, f"Multiple distributions overwrite the same Griffe modules: {overlaps}"


@pytest.fixture(scope="module")
def built_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output_dir = tmp_path_factory.mktemp("wheel")

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(output_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    wheels = list(output_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"
    return wheels[0]


def test_that_the_built_wheel_ships_the_widget_source_without_node_modules(
    built_wheel: Path,
) -> None:
    names = zipfile.ZipFile(built_wheel).namelist()

    # The vendorable widget and the playground harness both ship.
    assert "parlant/_widget/parlant-chat/index.ts" in names
    assert "parlant/_widget/parlant-chat/styles.css" in names
    assert "parlant/_widget/package.json" in names
    assert "parlant/_widget/sample-agent/agent.py" in names

    # The toolchain and any local runtime state do not.
    assert not [n for n in names if "node_modules" in n]
    assert not [n for n in names if "parlant-data" in n]
    assert not [n for n in names if n.startswith("parlant/_widget/dist/")]

    widget_bytes = sum(
        info.file_size
        for info in zipfile.ZipFile(built_wheel).infolist()
        if info.filename.startswith("parlant/_widget/")
    )
    widget_mb = widget_bytes / (1024 * 1024)
    assert widget_mb < MAX_WIDGET_MEGABYTES, f"the bundled widget is {widget_mb:.1f}MB"


def test_that_the_built_wheel_installs_the_parlant_console_script(
    built_wheel: Path,
) -> None:
    names = zipfile.ZipFile(built_wheel).namelist()

    entry_points = next(n for n in names if n.endswith(".dist-info/entry_points.txt"))
    content = zipfile.ZipFile(built_wheel).read(entry_points).decode()

    assert "[console_scripts]" in content
    assert "parlant" in content
