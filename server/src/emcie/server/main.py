from typing import Any, Dict
from fastapi import FastAPI

from emcie.server.api import agents
from emcie.server.api import threads
from emcie.server.agents import AgentStore
from emcie.server.models import ModelId, ModelRegistry
from emcie.server.providers.openai import GPT
from emcie.server.threads import ThreadStore


async def create_app(skills: Dict[str, Any] = {}) -> FastAPI:
    agent_store = AgentStore()
    thread_store = ThreadStore()
    model_registry = ModelRegistry()

    models = {
        "openai/gpt-4-turbo": GPT("gpt-4-turbo-preview"),
        "openai/gpt-3.5-turbo": GPT("gpt-3.5-turbo-0125"),
    }

    for model_id, model in models.items():
        await model_registry.add_text_generation_model(ModelId(model_id), model)

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

    return app
