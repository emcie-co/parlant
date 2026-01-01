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
import semver  # type: ignore
import sys
import subprocess
import toml  # type: ignore

from utils import die, for_each_package, Package, get_packages


def get_server_version() -> str:
    server_package = next(p for p in get_packages() if p.name == "parlant")
    project_file = server_package.path / "pyproject.toml"
    pyproject = toml.load(project_file)
    version = str(pyproject["tool"]["poetry"]["version"])
    return version


def run_command(args: list[str]) -> None:
    cmd = " ".join(args)

    print(f"Running {cmd}")

    build_process = subprocess.Popen(
        args=args,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    status = build_process.wait()

    if status != 0:
        die(f"error: command failed: {cmd}")


def publish_docker() -> None:
    version = get_server_version()
    version_info = semver.parse_version_info(version)

    tag_versions = [
        f"{version_info.major}.{version_info.minor}.{version_info.patch}.{version_info.prerelease}",
    ]

    if not version_info.prerelease:
        tag_versions = [
            "latest",
            f"{version_info.major}",
            f"{version_info.major}.{version_info.minor}",
            f"{version_info.major}.{version_info.minor}.{version_info.patch}",
        ]
    else:
        tag_versions = [
            f"{version_info.major}.{version_info.minor}.{version_info.patch}.{version_info.prerelease}",
        ]

    platforms = [
        "linux/amd64",
        "linux/arm64",
    ]

    for version in tag_versions:
        run_command(
            [
                "docker",
                "buildx",
                "build",
                "--platform",
                ",".join(platforms),
                "-t",
                f"ghcr.io/emcie-co/parlant:{version}",
                "-f",
                "Dockerfile",
                "--push",
                ".",
            ]
        )


def publish_package(package: Package) -> None:
    if not package.uses_uv or not package.publish:
        print(f"Skipping {package.path}...")
        return

    status, output = package.run_cmd("uv build")

    if status != 0:
        print(output, file=sys.stderr)
        die(f"error: package '{package.path}': build failed")

    status, output = package.run_cmd("uv publish")

    if status != 0:
        print(output, file=sys.stderr)
        die(f"error: package '{package.path}': publish failed")


if __name__ == "__main__":
    for_each_package(publish_package)
    publish_docker()
