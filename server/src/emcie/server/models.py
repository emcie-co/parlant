# Copyright (c) 2024 Emcie
# All rights reserved.
#
# This file and its contents are the property of Emcie and are strictly confidential.
# No part of this file may be reproduced, distributed, or transmitted in any form or by any means,
# including photocopying, recording, or other electronic or mechanical methods,
# without the prior written permission of Emcie.
#
# Website: https://emcie.co
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Iterable, NewType

from emcie.server.threads import Message


ModelId = NewType("ModelId", str)


class TextGenerationModel(ABC):
    @abstractmethod
    async def generate_text(
        self,
        messages: Iterable[Message],
        skills: Iterable[Any],
        rules: Iterable[Any],
    ) -> AsyncIterator[str]:
        yield ""


class TextEmbeddingModel(ABC):
    @abstractmethod
    def embed(
        self,
        text: str,
    ) -> Iterable[float]: ...


class ModelRegistry:
    def __init__(
        self,
    ) -> None:
        self._text_generation_models: Dict[ModelId, TextGenerationModel] = {}
        self._text_embedding_models: Dict[ModelId, TextEmbeddingModel] = {}

    async def add_text_generation_model(
        self,
        model_id: ModelId,
        model: TextGenerationModel,
    ) -> None:
        self._text_generation_models[model_id] = model

    async def get_text_generation_model(
        self,
        model_id: ModelId,
    ) -> TextGenerationModel:
        return self._text_generation_models[model_id]

    async def add_text_embedding_model(
        self,
        model_id: ModelId,
        model: TextGenerationModel,
    ) -> None:
        self._text_embedding_models[model_id] = model

    async def get_text_embedding_model(
        self,
        model_id: ModelId,
    ) -> TextGenerationModel:
        return self._text_embedding_models[model_id]
