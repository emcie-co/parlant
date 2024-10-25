from itertools import chain
import os
import httpx

from emcie.server.core.logging import Logger
from emcie.server.core.nlp.moderation import ModerationCheck, ModerationService, ModerationTag


class LakeraGuard(ModerationService):
    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._api_key = os.environ["LAKERA_API_KEY"]

    async def check(self, content: str) -> ModerationCheck:
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
                    headers={"Authorization": f"Bearer {self._api_key}"},
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
            flagged=any(detected for category, detected in results),
            tags=list(
                set(
                    chain.from_iterable(
                        extract_tags(category) for category, detected in results if detected
                    )
                )
            ),
        )
