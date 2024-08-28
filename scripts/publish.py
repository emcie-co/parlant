#!/usr/bin/python3
import semver
import sys
import subprocess
import toml  # type: ignore
from utils import die, for_each_package, Package, get_packages


def get_server_version() -> str:
    server_package = next(p for p in get_packages() if p.name == "server")
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
        f"v{version_info.major}",
        f"v{version_info.major}.{version_info.minor}",
        f"v{version_info.major}.{version_info.minor}.{version_info.patch}",
        f"v{version_info.major}.{version_info.minor}.{version_info.patch}.{version_info.prerelease}",
    ]

    if not version_info.prerelease:
        tag_versions.append("latest")

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
                f"ghcr.io/emcie-co/emcie-server:{version}",
                "-f",
                "Dockerfile.server",
                "--push",
                ".",
            ]
        )


def publish_package(package: Package) -> None:
    if not package.uses_poetry or not package.publish:
        print(f"Skipping {package.path}...")
        return

    status, output = package.run_cmd("poetry build")

    if status != 0:
        print(output, file=sys.stderr)
        die(f"error: package '{package.path}': build failed")

    status, output = package.run_cmd("poetry publish")

    if status != 0:
        print(output, file=sys.stderr)
        die(f"error: package '{package.path}': publish failed")


if __name__ == "__main__":
    for_each_package(publish_package)
    publish_docker()
