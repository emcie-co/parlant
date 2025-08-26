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

from itertools import chain
import os
from typing_extensions import override
import httpx

from parlant.core.loggers import Logger
from parlant.core.nlp.moderation import ModerationCheck, ModerationService, ModerationTag


class LakeraGuard(ModerationService):
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    @override
    async def check(self, content: str) -> ModerationCheck:
        api_key: str | None = os.environ.get("LAKERA_API_KEY")

        if not api_key:
            self._logger.warning(
                "LakeraGuard is enabled but LAKERA_API_KEY is missing. Skipping check..."
            )
            return ModerationCheck(flagged=False, tags=[])

        def extract_tags(category: str) -> list[ModerationTag]:
            mapping: dict[str, list[ModerationTag]] = {
                "moderated_content_crime": ["illicit"],
                "moderated_content_hate": ["hate"],
                "moderated_content_profanity": ["harassment"],
                "moderated_content_sexual": ["sexual"],
                "moderated_content_violence": ["violence"],
                "prompt_attack": ["jailbreak"],
            }

            return mapping.get(category.replace("/", "_").replace("-", "_"), [])

        with self._logger.operation("Lakera Moderation Request"):
            async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                response = await client.post(
                    "https://api.lakera.ai/v2/guard/results",
                    json={"messages": [{"content": content, "role": "user"}]},
                    headers={"Authorization": f"Bearer {api_key}"},
                )

                if response.is_error:
                    raise Exception("Moderation service failure (Lakera Guard)")

                data = response.json()

        results = [
            (
                r["detector_type"],
                {
                    "l1_confident": True,
                    "l2_very_likely": True,
                    "l3_likely": True,
                    "l4_less_likely": False,
                    "l5_unlikely": False,
                }.get(r["result"], False),
            )
            for r in data["results"]
        ]

        return ModerationCheck(
            flagged=any(detected for _category, detected in results),
            tags=list(
                set(
                    chain.from_iterable(
                        extract_tags(category) for category, detected in results if detected
                    )
                )
            ),
        )
