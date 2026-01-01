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

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, NoReturn


@dataclass(frozen=True)
class Package:
    name: str
    path: Path
    uses_uv: bool
    cmd_prefix: str
    publish: bool

    def run_cmd(self, cmd: str) -> tuple[int, str]:
        print(f"Running command: {self.cmd_prefix} {cmd}")
        return subprocess.getstatusoutput(f"{self.cmd_prefix} {cmd}")


def get_repo_root() -> Path:
    status, output = subprocess.getstatusoutput("git rev-parse --show-toplevel")

    if status != 0:
        print(output, file=sys.stderr)
        print("error: failed to get repo root", file=sys.stderr)
        sys.exit(1)

    return Path(output.strip())


def get_packages() -> list[Package]:
    root = get_repo_root()

    return [
        Package(
            name="parlant",
            path=root / ".",
            cmd_prefix="uv run",
            uses_uv=True,
            publish=True,
        ),
    ]


def for_each_package(
    f: Callable[[Package], None],
    enter_dir: bool = True,
) -> None:
    for package in get_packages():
        original_cwd = os.getcwd()

        if enter_dir:
            print(f"Entering {package.path}...")
            os.chdir(package.path)

        try:
            f(package)
        finally:
            os.chdir(original_cwd)


def die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    sys.exit(1)
