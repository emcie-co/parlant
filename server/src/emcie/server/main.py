# Copyright (c) 2024 Emcie
# All rights reserved.
#
# This file and its contents are the property of Emcie and are strictly confidential.
# No part of this file may be reproduced, distributed, or transmitted in any form or by any means,
# including photocopying, recording, or other electronic or mechanical methods,
# without the prior written permission of Emcie.
#
# Website: https://emcie.co

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI

from emcie.server.api import agents
from emcie.server.api import threads
from emcie.server.api import rag

from emcie.server.agents import AgentStore
from emcie.server.models import ModelId, ModelRegistry, TextEmbeddingModel, TextGenerationModel
from emcie.server.providers.openai import (
    AzureTextEmbedding,
    OpenAIGPT,
    AzureGPT,
    OpenAITextEmbedding,
)
from emcie.server.rag import RagStore
from emcie.server.threads import ThreadStore

from emcie.server.models import ModelId, ModelRegistry


async def create_app(
    agent_store: Optional[AgentStore] = None,
    thread_store: Optional[ThreadStore] = None,
    rag_store: Optional[RagStore] = None,
    skills: Dict[str, Any] = {},
    rules: List[Any] = [],
) -> FastAPI:
    agent_store = agent_store or AgentStore()
    thread_store = thread_store or ThreadStore()
    model_registry = ModelRegistry()

    models = {
        "openai/gpt-4-turbo": OpenAIGPT("gpt-4-turbo-preview"),
        "openai/gpt-3.5-turbo": OpenAIGPT("gpt-3.5-turbo-0125"),
        "openai/text-embedding-ada-002": OpenAITextEmbedding("text-embedding-ada-002"),
        "azure/gpt-3.5-turbo": AzureGPT("gpt-35-turbo"),
        "azure/gpt-4": AzureGPT("gpt-4"),
        "azure/text-embedding-ada-002": AzureTextEmbedding("text-embedding-ada-002"),
    }

    for model_id, model in models.items():
        if isinstance(model, TextGenerationModel):
            await model_registry.add_text_generation_model(ModelId(model_id), model)
        elif isinstance(model, TextEmbeddingModel):
            await model_registry.add_text_embedding_model(ModelId(model_id), model)

    rag_store = rag_store or RagStore(
        embedding_model=await model_registry.get_text_embedding_model(
            os.environ["DEFAULT_RAG_MODEL"],
        )
    )

    for skill_id, skill in skills.items():
        await agent_store.create_skill(
            skill_id=skill_id,
            module_path=skill["module_path"],
            spec={
                "type": "function",
                "function": {
                    "name": skill["name"],
                    "description": skill["description"],
                    "parameters": {
                        "type": "object",
                        "properties": skill["parameters"],
                    },
                    "required": skill["required"],
                },
            },
        )

    for rule in rules:
        await agent_store.create_rule(
            when=rule["when"],
            then=rule["then"],
        )

    app = FastAPI()

    app.include_router(
        prefix="/agents",
        router=agents.create_router(
            agent_store=agent_store,
            thread_store=thread_store,
            model_registry=model_registry,
        ),
    )

    app.mount(
        "/threads",
        threads.create_router(
            thread_store=thread_store,
        ),
    )

    app.mount(
        "/rag",
        rag.create_router(
            rag_store=rag_store,
        ),
    )

    return app
