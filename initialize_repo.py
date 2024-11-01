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
