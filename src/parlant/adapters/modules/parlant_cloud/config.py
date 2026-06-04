# Copyright 2026 Parlant (Emcie Co Ltd.)
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

"""Shared configuration and constants for the Parlant Cloud module."""

import logging
import os

_logger = logging.getLogger(__name__)

PROJECT_TOKEN_HEADER = "X-Parlant-Cloud-Project-Token"

_DEFAULT_BASE_URL = "https://api.parlant.cloud"


def _get_cloud_base_url() -> str:
    """Resolve the Parlant Cloud base URL from environment.

    Priority: PARLANT_CLOUD_BASE_URL > PARLANT_CLOUD_OTEL_URL > default.
    """
    return (
        os.getenv("PARLANT_CLOUD_BASE_URL")
        or os.getenv("PARLANT_CLOUD_OTEL_URL")
        or _DEFAULT_BASE_URL
    ).rstrip("/")
