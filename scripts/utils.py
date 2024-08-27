from dataclasses import field, dataclass
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, NoReturn


@dataclass(frozen=True)
class Package:
    name: str
    path: Path
    uses_poetry: bool
    cmd_prefix: str
    bin_files: list[Path] = field(default_factory=list)

    def run_cmd(self, cmd: str) -> tuple[int, str]:
        return subprocess.getstatusoutput(f"{self.cmd_prefix} {cmd}")


def get_repo_root() -> Path:
    status, output = subprocess.getstatusoutput("git rev-parse --show-toplevel")

    if status != 0:
        print(output, file=sys.stderr)
        print("error: failed to get repo root", file=sys.stderr)
        exit(1)

    return Path(output.strip())


def get_packages() -> list[Package]:
    root = get_repo_root()

    return [
        Package(
            name="scripts",
            path=root / "scripts",
            cmd_prefix="",
            uses_poetry=False,
        ),
        Package(
            name="common",
            path=root / "common",
            cmd_prefix="poetry run",
            uses_poetry=True,
        ),
        Package(
            name="sdk",
            path=root / "sdk",
            cmd_prefix="poetry run",
            uses_poetry=True,
        ),
        Package(
            name="server",
            path=root / "server",
            cmd_prefix="poetry run",
            bin_files=[
                root / "server" / "bin" / "emcie",
                root / "server" / "bin" / "emcie-server",
            ],
            uses_poetry=True,
        ),
    ]


def for_each_package(
    f: Callable[[Package], None],
    enter_dir: bool = True,
) -> None:
    for package in get_packages():
        if enter_dir:
            print(f"Entering {package.path}...")
            os.chdir(package.path)

        f(package)


def die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    exit(1)