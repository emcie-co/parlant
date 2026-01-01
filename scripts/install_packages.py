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

import subprocess
import sys

from utils import Package, die, for_each_package


def install_package(package: Package) -> None:
    if not package.uses_uv:
        print(f"Skipping {package.path}...")
        return

    print(f"Installing {package.path}...")

    status, output = subprocess.getstatusoutput(f"uv sync --all-extras --directory {package.path}")

    if status != 0:
        print(output, file=sys.stderr)
        die(f"error: failed to install package: {package.path}")


if __name__ == "__main__":
    for_each_package(install_package)
