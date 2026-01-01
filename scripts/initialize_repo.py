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
from pathlib import Path

SCRIPTS_DIR = Path("./scripts")


def install_packages() -> None:
    subprocess.run(["python", SCRIPTS_DIR / "install_packages.py"])


def install_hooks() -> None:
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], check=True)


if __name__ == "__main__":
    install_packages()
    install_hooks()
