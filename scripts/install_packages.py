import subprocess
import sys
from utils import Package, die, for_each_package


def install_package(package: Package) -> None:
    if not package.uses_poetry:
        print(f"Skipping {package.path}...")
        return

    print(f"Installing {package.path}...")

    status, output = subprocess.getstatusoutput(f"poetry -C {package.path} install")

    if status != 0:
        print(output, file=sys.stderr)
        die(f"error: failed to install package: {package.path}")


if __name__ == "__main__":
    for_each_package(install_package)
