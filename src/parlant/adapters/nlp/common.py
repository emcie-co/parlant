# Copyright 2025 Emcie Co Ltd.
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


def normalize_json_output(raw_output: str) -> str:
    json_start = raw_output.find("```json")

    if json_start != -1:
        json_start = json_start + 7
    else:
        json_start = 0

    json_end = raw_output[json_start:].rfind("```")

    if json_end == -1:
        json_end = len(raw_output[json_start:])

    return raw_output[json_start : json_start + json_end].strip()
