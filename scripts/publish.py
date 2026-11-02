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
"""Builds and publishes the package.

Every build runs in a throwaway git worktree of HEAD, so an artifact holds only
committed files. Every artifact is then inspected against `DENIED`, and one
match aborts the release before anything is uploaded.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import semver  # type: ignore
import toml  # type: ignore

from utils import die, for_each_package, get_packages, get_repo_root, Package


# Paths that must never appear inside a published artifact: a label and a
# pattern, matched against every member of the wheel and the sdist.
DENIED: list[tuple[str, str]] = [
    # Not part of this distribution.
    ("pro sources", r"(^|/)pro/"),
    ("pro package", r"parlant[-_/]pro(/|\.|$)"),
    # Whatever a running agent wrote locally.
    ("runtime data", r"(^|/)parlant-data(/|$)"),
    ("databases", r"\.(db|sqlite|sqlite3)$"),
    ("logs", r"\.log$"),
    # A real .env or a variant like .env.local, but not the .env.example the
    # widget ships as a template.
    ("environment files", r"(^|/)\.env(\.(?!example|template|sample)[^/]*)?$"),
    ("node modules", r"(^|/)node_modules(/|$)"),
    # Build output, which the consumer rebuilds from the source that ships.
    ("widget build output", r"(^|/)widget/dist(/|$)|_widget/dist(/|$)"),
    ("typescript build info", r"\.tsbuildinfo$"),
]


def get_server_version() -> str:
    server_package = next(p for p in get_packages() if p.name == "parlant")
    pyproject = toml.load(server_package.path / "pyproject.toml")
    return str(pyproject["project"]["version"])


def run_command(
    args: Sequence[str],
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    cmd = " ".join(args)

    print(f"Running {cmd}" + (f" (in {cwd})" if cwd else ""))

    process = subprocess.Popen(
        args=list(args),
        cwd=cwd,
        # Secrets travel here, not in `args`, which is world-readable.
        env={**os.environ, **(env or {})},
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    if process.wait() != 0:
        die(f"error: command failed: {cmd}")


def capture(args: Sequence[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        args=list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        die(f"error: command failed: {' '.join(args)}")

    return result.stdout.strip()


def require_committed_tree(root: Path) -> str:
    """Return the commit to build, refusing when tracked files differ from it."""

    modified = capture(["git", "diff", "--name-only", "HEAD"], cwd=root)

    if modified:
        print(
            "error: tracked files differ from HEAD, so the build would omit them:", file=sys.stderr
        )
        for path in modified.splitlines():
            print(f"  {path}", file=sys.stderr)
        die("Commit or stash them, then publish.")

    return capture(["git", "rev-parse", "HEAD"], cwd=root)


def require_pypi_token() -> str:
    """The PyPI token, from $PYPI_TOKEN. Checked before anything is built."""

    token = os.environ.get("PYPI_TOKEN", "").strip()

    if not token:
        die(
            "error: PYPI_TOKEN is not set.\n"
            "Export it in your shell (it is never read from a file or logged):\n"
            "  export PYPI_TOKEN=pypi-..."
        )

    return token


def artifact_members(artifact: Path) -> Iterable[str]:
    if artifact.suffix == ".whl":
        with zipfile.ZipFile(artifact) as wheel:
            yield from wheel.namelist()
    elif artifact.name.endswith(".tar.gz"):
        with tarfile.open(artifact) as sdist:
            yield from sdist.getnames()
    else:
        die(f"error: don't know how to inspect '{artifact.name}'")


def verify_artifact(artifact: Path) -> None:
    """Abort the release if the artifact carries anything from DENIED."""

    members = list(artifact_members(artifact))

    if not members:
        die(f"error: '{artifact.name}' is empty")

    offenders: list[tuple[str, str]] = []

    for label, pattern in DENIED:
        expression = re.compile(pattern)
        for member in members:
            if expression.search(member):
                offenders.append((label, member))

    if offenders:
        print(f"error: '{artifact.name}' must not be published:", file=sys.stderr)
        for label, member in offenders[:40]:
            print(f"  {label}: {member}", file=sys.stderr)
        if len(offenders) > 40:
            print(f"  ... and {len(offenders) - 40} more", file=sys.stderr)
        die("Nothing was uploaded. Check the hatch build targets in pyproject.toml.")

    print(f"  {artifact.name}: {len(members)} entries, nothing denied")


def build_in_clean_worktree(package: Package, commit: str) -> list[Path]:
    """Build from a fresh checkout of `commit`, holding only committed files."""

    root = get_repo_root()
    version = get_server_version()
    output_dir = root / "dist" / version

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    with tempfile.TemporaryDirectory(prefix="parlant-release-") as scratch:
        tree = Path(scratch) / "source"

        run_command(["git", "worktree", "add", "--detach", str(tree), commit], cwd=root)

        try:
            leftovers = capture(["git", "status", "--porcelain"], cwd=tree)

            if leftovers:
                die(f"error: the build worktree is not clean:\n{leftovers}")

            run_command(["uv", "build", "--out-dir", str(output_dir)], cwd=tree)
        finally:
            run_command(["git", "worktree", "remove", "--force", str(tree)], cwd=root)

    # uv drops its own .gitignore beside the artifacts; publish distributions only.
    artifacts = sorted(
        path
        for path in output_dir.iterdir()
        if path.suffix == ".whl" or path.name.endswith(".tar.gz")
    )

    if not artifacts:
        die(f"error: package '{package.path}': the build produced nothing")

    return artifacts


def publish_docker() -> None:
    """Build and push the container images. Requires --docker."""

    version = get_server_version()
    version_info = semver.parse_version_info(version)

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

    token = require_pypi_token()
    commit = require_committed_tree(get_repo_root())

    print(f"Building {package.name} from a clean worktree of {commit[:9]}...")
    artifacts = build_in_clean_worktree(package, commit)

    print("Checking what the build produced...")
    for artifact in artifacts:
        verify_artifact(artifact)

    run_command(
        ["uv", "publish", *(str(a) for a in artifacts)],
        env={"UV_PUBLISH_TOKEN": token},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish the package to PyPI.")
    parser.add_argument(
        "--docker",
        action="store_true",
        help="also build and push the container images to ghcr.io (not done by default)",
    )
    arguments = parser.parse_args()

    for_each_package(publish_package)

    if arguments.docker:
        publish_docker()


if __name__ == "__main__":
    main()
