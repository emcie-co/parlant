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

#!/usr/bin/python3
from functools import partial
from pathlib import Path
import semver  # type: ignore
import subprocess
import sys
import re
import toml  # type: ignore

from utils import die, for_each_package, Package, get_packages


def get_project_file(package: Package) -> Path:
    return package.path / "pyproject.toml"


def get_current_version(package: Package) -> str:
    content = toml.load(get_project_file(package))
    return str(content["project"]["version"])


def set_package_version(version: str, package: Package) -> None:
    if not package.uses_uv:
        print(f"Skipping {package.path}...")
        return

    current_version = get_current_version(package)

    print(f"Setting {package.name} from version {current_version} to version {version}")

    project_file = get_project_file(package)

    project_file_content = project_file.read_text()

    with open(project_file, "w") as file:
        project_file_content = re.sub(
            f'\nversion = "{current_version}"\n',
            f'\nversion = "{version}"\n',
            project_file_content,
            count=1,
        )

        project_file_content = re.sub(
            f'\nparlant-(.+?) = "{current_version}"\n',
            f'\nparlant-\\1 = "{version}"\n',
            project_file_content,
        )

        file.write(project_file_content)

    status, output = package.run_cmd("uv lock")

    if status != 0:
        print(output, file=sys.stderr)
        die("error: failed to re-hash uv lock file")


def update_version_variable_in_code(version: str) -> None:
    server_package = next(p for p in get_packages() if p.name == "parlant")
    version_file: Path = server_package.path / "src/parlant/core/version.py"

    version_file_content = version_file.read_text()
    current_version = get_current_version(server_package)

    version_file_content = re.sub(
        f'VERSION = "{current_version}"',
        f'VERSION = "{version}"',
        version_file_content,
    )

    version_file.write_text(version_file_content)


def tag_repo(version: str) -> None:
    status, output = subprocess.getstatusoutput(f'git tag "v{version}"')

    if status != 0:
        print(output, file=sys.stderr)
        die(f"error: failed to tag repo: v{version}")


def get_current_server_version() -> str:
    server_package = next(p for p in get_packages() if p.name == "parlant")
    return get_current_version(server_package)


def update_version(
    current_version: str,
    major: bool,
    minor: bool,
    patch: bool,
    rc: bool,
    beta: bool,
    alpha: bool,
) -> str:
    assert sum((major, minor, patch)) <= 1, "Only one component can be bumped"
    assert sum((rc, beta, alpha)) <= 1, "Only one pre-release label can be used"

    version = semver.parse_version_info(current_version)

    if major:
        version = version.bump_major()
    if minor:
        version = version.bump_minor()
    if patch:
        version = version.bump_patch()

    if rc:
        version = version.bump_prerelease("rc")
    elif beta:
        version = version.bump_prerelease("beta")
    elif alpha:
        version = version.bump_prerelease("alpha")
    else:
        version = version.finalize_version()

    return str(version)


def there_are_pending_git_changes() -> bool:
    status, _ = subprocess.getstatusoutput(
        "git diff --quiet && git diff --cached --quiet && git ls-files --others --exclude-standard"
    )
    return status != 0


def commit_version(version: str) -> bool:
    status, _ = subprocess.getstatusoutput(f"git commit -am 'Release {version}' --no-verify")
    return status != 0


if __name__ == "__main__":
    if there_are_pending_git_changes():
        die("error: version bumps must take place on a clean tree with no pending changes")

    current_version = get_current_server_version()

    major = "--major" in sys.argv
    minor = "--minor" in sys.argv
    patch = "--patch" in sys.argv
    rc = "--rc" in sys.argv
    beta = "--beta" in sys.argv
    alpha = "--alpha" in sys.argv

    new_version = update_version(current_version, major, minor, patch, rc, beta, alpha)

    if current_version == new_version:
        die("error: no component was selected to be bumped")

    answer = input(f"Proceed with bumping {current_version} to {new_version} [N/y]?")

    if answer not in "yY":
        die("Canceled.")

    update_version_variable_in_code(new_version)
    for_each_package(partial(set_package_version, new_version))
    commit_version(new_version)
    tag_repo(new_version)
