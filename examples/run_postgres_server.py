#!/usr/bin/env python
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

"""
Start Parlant server with PostgreSQL storage and OpenAI NLP service.

Required environment variables (can be set in .env):
    POSTGRES_CONNECTION_STRING  - PostgreSQL connection string
                                  (e.g., postgresql://user:pass@localhost:5432/parlant)
    OPENAI_API_KEY              - OpenAI API key

Optional environment variables:
    PARLANT_HOST - Server host (default: 0.0.0.0)
    PARLANT_PORT - Server port (default: 8800)

Usage:
    uv run python examples/run_postgres_server.py
"""

import asyncio
import os
import sys
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from lagom import Container


async def run() -> None:
    postgres_url = os.environ.get("POSTGRES_CONNECTION_STRING")
    if not postgres_url:
        print("Error: POSTGRES_CONNECTION_STRING environment variable is required")
        print("Example: postgresql://user:password@localhost:5432/parlant")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    from parlant.adapters.db.json_file import JSONFileDocumentDatabase
    from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase
    from parlant.core.agents import AgentDocumentStore, AgentStore
    from parlant.core.canned_responses import CannedResponseStore, CannedResponseVectorStore
    from parlant.core.capabilities import CapabilityStore, CapabilityVectorStore
    from parlant.bin.server import PARLANT_HOME_DIR
    from parlant.core.common import IdGenerator
    from parlant.core.emission.event_publisher import EventEmitterFactory, EventPublisherFactory
    from parlant.core.services.tools.service_registry import (
        ServiceDocumentRegistry,
        ServiceRegistry,
    )
    from parlant.core.evaluations import EvaluationDocumentStore, EvaluationStore
    from parlant.core.glossary import GlossaryStore, GlossaryVectorStore
    from parlant.core.guideline_tool_associations import (
        GuidelineToolAssociationDocumentStore,
        GuidelineToolAssociationStore,
    )
    from parlant.core.guidelines import GuidelineDocumentStore, GuidelineStore
    from parlant.core.journeys import JourneyStore, JourneyVectorStore
    from parlant.core.loggers import Logger
    from parlant.core.nlp.embedding import EmbedderFactory, EmbeddingCache
    from parlant.core.nlp.service import NLPService
    from parlant.core.relationships import RelationshipDocumentStore, RelationshipStore
    from parlant.core.sessions import SessionStore
    from parlant.core.tags import TagDocumentStore, TagStore
    from parlant.core.test_suites import TestSuiteDocumentStore, TestSuiteStore
    from parlant.core.tracer import Tracer
    from parlant.sdk import NLPServices, Server

    host = os.environ.get("PARLANT_HOST", "0.0.0.0")
    port = int(os.environ.get("PARLANT_PORT", "8800"))

    exit_stack = AsyncExitStack()

    async def configure_json_stores(container: Container) -> Container:
        """Override transient stores with JSON-backed stores for persistence."""
        logger = container[Logger]
        id_generator = container[IdGenerator]

        # Document stores - persist to JSON files
        for interface, implementation, filename in [
            (AgentStore, AgentDocumentStore, "agents.json"),
            (TagStore, TagDocumentStore, "tags.json"),
            (GuidelineStore, GuidelineDocumentStore, "guidelines.json"),
            (
                GuidelineToolAssociationStore,
                GuidelineToolAssociationDocumentStore,
                "guideline_tool_associations.json",
            ),
            (RelationshipStore, RelationshipDocumentStore, "relationships.json"),
            (TestSuiteStore, TestSuiteDocumentStore, "test_suites.json"),
        ]:
            db = await exit_stack.enter_async_context(
                JSONFileDocumentDatabase(logger, PARLANT_HOME_DIR / filename)
            )
            container[interface] = await exit_stack.enter_async_context(
                implementation(id_generator, db)
            )

        # Evaluation store (no id_generator)
        eval_db = await exit_stack.enter_async_context(
            JSONFileDocumentDatabase(logger, PARLANT_HOME_DIR / "evaluations.json")
        )
        container[EvaluationStore] = await exit_stack.enter_async_context(
            EvaluationDocumentStore(eval_db)
        )

        # Vector stores - persist to pgvector + JSON
        tracer = container[Tracer]
        embedder_factory = EmbedderFactory(container)

        async def get_embedder_type() -> type:
            return type(await container[NLPService].get_embedder())

        # Create a single pgvector database instance for all vector stores
        vector_db = await exit_stack.enter_async_context(
            PostgresVectorDatabase(
                connection_string=postgres_url,
                logger=logger,
                tracer=tracer,
                embedder_factory=embedder_factory,
                embedding_cache_provider=lambda: container[EmbeddingCache],
            )
        )

        for vector_store_interface, vector_store_type, filename in [
            (GlossaryStore, GlossaryVectorStore, "glossary_tags.json"),
            (CannedResponseStore, CannedResponseVectorStore, "canned_responses.json"),
            (JourneyStore, JourneyVectorStore, "journey_associations.json"),
            (CapabilityStore, CapabilityVectorStore, "capabilities.json"),
        ]:
            doc_db = await exit_stack.enter_async_context(
                JSONFileDocumentDatabase(logger, PARLANT_HOME_DIR / filename)
            )
            container[vector_store_interface] = await exit_stack.enter_async_context(
                vector_store_type(
                    id_generator=id_generator,
                    vector_db=vector_db,
                    document_db=doc_db,
                    embedder_factory=embedder_factory,
                    embedder_type_provider=get_embedder_type,
                )
            )

        # Recreate EventEmitterFactory with the updated AgentStore
        # (the SDK creates it with transient stores before our callback runs)
        container[EventEmitterFactory] = EventPublisherFactory(
            agent_store=container[AgentStore],
            session_store=container[SessionStore],
        )

        # Recreate ServiceRegistry with persistent storage for tool services
        services_db = await exit_stack.enter_async_context(
            JSONFileDocumentDatabase(logger, PARLANT_HOME_DIR / "services.json")
        )
        container[ServiceRegistry] = await exit_stack.enter_async_context(
            ServiceDocumentRegistry(
                database=services_db,
                event_emitter_factory=container[EventEmitterFactory],
                logger=logger,
                tracer=tracer,
                nlp_services_provider=lambda: {"__nlp__": container[NLPService]},
                allow_migration=True,
            )
        )

        return container

    async with exit_stack:
        async with Server(
            host=host,
            port=port,
            nlp_service=NLPServices.openai,
            session_store=postgres_url,
            customer_store=postgres_url,
            variable_store=postgres_url,
            migrate=True,
            configure_container=configure_json_stores,
        ):
            pass


def main() -> None:
    load_dotenv()
    asyncio.run(run())


if __name__ == "__main__":
    main()
