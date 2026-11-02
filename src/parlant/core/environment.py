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

"""The environment a Parlant server runs in.

Set with ``PARLANT_ENV``. The environment decides how the server configures
itself — most visibly its authorization policy — and is recorded on every
session it creates, so observability can tell real traffic apart from a
developer poking at their agent.
"""

from enum import Enum
import os

ENVIRONMENT_ENV_VAR = "PARLANT_ENV"


class Environment(Enum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


def resolve_environment(value: str | None = None) -> Environment:
    """The configured environment, defaulting to development.

    Anything unrecognized is development: the modes that relax protections
    must be chosen deliberately, never fallen into by a typo.
    """
    raw = (value if value is not None else os.environ.get(ENVIRONMENT_ENV_VAR, "")).strip().lower()

    try:
        return Environment(raw)
    except ValueError:
        return Environment.DEVELOPMENT
