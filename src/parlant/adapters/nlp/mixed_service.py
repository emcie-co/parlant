# Copyright 2025  Emcie Co Ltd.
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

# Maintainer: Agam Dubey <hello.world.agam@gmail.com>

from typing_extensions import override
from parlant.adapters.nlp.ollama_service import OllamaService
from parlant.adapters.nlp.openai_service import OpenAIService


from parlant.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_batch import (
    GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
)
from parlant.core.engines.alpha.guideline_matching.generic.guideline_previously_applied_actionable_customer_dependent_batch import (
    GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
)
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import Embedder
from parlant.core.nlp.generation import (
    T,
    SchematicGenerator,
)
from parlant.core.loggers import Logger


class MixedService(NLPService):
    """NLP Service that mixes Ollama and OPENAI."""

    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""
        OllamaService.verify_environment()
        OpenAIService.verify_environment()

    @staticmethod
    def verify_models() -> str | None:
        """
        Verify that the required models are available in Ollama.
        Returns an error message if models are missing, None if all are available.
        """
        OllamaService.verify_models()

    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self.openai_service = OpenAIService(logger)
        self.ollama_service = OllamaService(logger)

    @override
    async def get_schematic_generator(self, t: type[T]) -> SchematicGenerator[T]:
        """Get a schematic generator for the specified type."""
        if t in {
            GenericPreviouslyAppliedActionableGuidelineMatchesSchema,
            GenericPreviouslyAppliedActionableCustomerDependentGuidelineMatchesSchema,
        }:
            return await self.ollama_service.get_schematic_generator(t)
        return await self.openai_service.get_schematic_generator(t)

    @override
    async def get_embedder(self) -> Embedder:
        return await self.openai_service.get_embedder()

    @override
    async def get_moderation_service(self) -> ModerationService:
        """Get a moderation service (using no moderation for local models)."""
        return NoModeration()
