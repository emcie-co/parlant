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

#!python

import os
from pathlib import Path
import re
import subprocess
import shutil
import sys
import time


DIR_SCRIPT_ROOT = Path(__file__).parent
DIR_FERN = DIR_SCRIPT_ROOT / "fern"
DIR_SDKS = DIR_SCRIPT_ROOT / "sdks"
DIR_PROJECTS_WORKSPACE = DIR_SCRIPT_ROOT / ".." / ".." / "parlant-sdks"


PATHDICT_SDK_REPO_TARGETS = {
    "python": DIR_PROJECTS_WORKSPACE / "parlant-client-python" / "src" / "parlant" / "client",
    "typescript": DIR_PROJECTS_WORKSPACE / "parlant-client-typescript" / "src",
}


def replace_in_files(rootdir: Path, search: str, replace: str) -> None:
    rewrites: dict[str, str] = {}
    for subdir, _dirs, files in os.walk(rootdir):
        for file in files:
            file_path = os.path.join(subdir, file)

            with open(file_path, "r") as current_file:
                current_file_content = current_file.read()
                if "from parlant import" not in current_file_content:
                    continue

                current_file_content = re.sub(search, replace, current_file_content)
                rewrites[file_path] = current_file_content

    for path, content in rewrites.items():
        with open(path, "w") as current_file:
            current_file.write(content)


if __name__ == "__main__":
    DEFAULT_PORT = 8800
    port = DEFAULT_PORT
    if len(sys.argv) >= 2:
        port = int(sys.argv[1])

    print(f"The script will now try to fetch the latest openapi.json from http://localhost:{port}.")
    input(
        f"Ensure that parlant-server is running on port {port} and then press any key to continue..."
    )

    output_openapi_json = DIR_FERN / "openapi/parlant.openapi.json"
    output_openapi_json.parent.mkdir(exist_ok=True)
    output_openapi_json.touch()

    status, output = subprocess.getstatusoutput(
        f"curl -m 3 -o {output_openapi_json} http://localhost:{port}/openapi.json"
    )

    if status != 0:
        print(f"Failed to fetch openapi.json from http://localhost:{port}", file=sys.stderr)
        print("Please ensure that the desired Parlant server is accessible there.", file=sys.stderr)
        sys.exit(1)

    for sdk, repo in PATHDICT_SDK_REPO_TARGETS.items():
        if os.path.isdir(repo):
            continue

        raise Exception(f"Missing dir for {sdk}: {repo}")

    print(f"Fetched openapi.json from http://localhost:{port}.")

    if not DIR_FERN.is_dir():
        raise Exception("fern directory not found where expected")
    for sdk in PATHDICT_SDK_REPO_TARGETS:
        sdk_path = DIR_SDKS / sdk
        if not sdk_path.is_dir():
            continue

        print(f"Deleting old {sdk} sdk")
        print(f"> rm -rf {sdk_path}")
        shutil.rmtree(sdk_path)

    os.chdir(DIR_SCRIPT_ROOT)

    print("Invoking fern generation")
    print("> fern generate --log-level=debug")
    exit_code, generate_output = subprocess.getstatusoutput("fern generate --log-level=debug")
    with open("fern.generate.log", "w") as fern_log:
        fern_log.write(generate_output)
    if exit_code != os.EX_OK:
        raise Exception(generate_output)

    print("Renaming `parlant` to `parlant.client` in python imports")
    replace_in_files(DIR_SDKS / "python", "from parlant import", "from parlant.client import")

    print("touching python typing")

    print(f"> touch {DIR_SDKS}/python/py.typed")
    open(DIR_SDKS / "python/py.typed", "w")

    for sdk, repo in PATHDICT_SDK_REPO_TARGETS.items():
        print(f"!DANGER! Deleting local `{repo}` directory and all of its contents!")
        time.sleep(3)
        print(f"> rm -rf {repo}")
        shutil.rmtree(repo)

    for sdk, repo in PATHDICT_SDK_REPO_TARGETS.items():
        print(f"copying newly generated {sdk} files to {repo}")
        print(f"> cp -rp {DIR_SDKS}/{sdk} {repo}")
        shutil.copytree(DIR_SDKS / sdk, repo)
