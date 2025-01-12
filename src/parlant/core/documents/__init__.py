# Copyright 2024 Emcie Co Ltd.
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

import re
from typing import Any

from parlant.core.common import SCHEMA_VERSION_UNKNOWN, SchemaVersion


def get_schema_version(cls: Any) -> SchemaVersion:
    """Extract schema version from TypedDict class name with _v123 suffix."""

    version_match = re.search("^.+_v(\\d+)$", cls.__name__)
    if version_match and version_match.lastindex:
        return SchemaVersion(int(version_match[version_match.lastindex]))
    return SCHEMA_VERSION_UNKNOWN
