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

import asyncio
from contextlib import AsyncExitStack, suppress
from datetime import datetime, timezone
import importlib
import json
import os
import shutil
from typing import Any, cast, Callable, Awaitable, Optional
import chromadb
from chromadb.api.types import IncludeEnum
from lagom import Container
from typing_extensions import NoReturn
from pathlib import Path
import sys
import rich
from rich.prompt import Confirm, Prompt

from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.adapters.vector_db.chroma import ChromaDatabase
from parlant.core.common import generate_id, md5_checksum, Version
from parlant.core.context_variables import (
    _ContextVariableDocument_v0_1_0,
    _ContextVariableTagAssociationDocument,
    ContextVariableId,
)
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.glossary import (
    _TermDocument_v0_1_0,
    _TermTagAssociationDocument,
    TermId,
)
from parlant.core.guidelines import (
    _GuidelineDocument_v0_2_0,
    _GuidelineTagAssociationDocument,
    _GuidelineDocument,
    GuidelineId,
    guideline_document_converter_0_1_0_to_0_2_0,
    _GuidelineDocument_v0_1_0,
)
from parlant.core.loggers import LogLevel, StdoutLogger
from parlant.core.nlp.embedding import EmbedderFactory
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    identity_loader,
)
from parlant.core.persistence.document_database_helper import _MetadataDocument
from parlant.core.tags import Tag

DEFAULT_HOME_DIR = "runtime-data" if Path("runtime-data").exists() else "parlant-data"
PARLANT_HOME_DIR = Path(os.environ.get("PARLANT_HOME", DEFAULT_HOME_DIR))
PARLANT_HOME_DIR.mkdir(parents=True, exist_ok=True)

EXIT_STACK = AsyncExitStack()

sys.path.append(PARLANT_HOME_DIR.as_posix())
sys.path.append(".")

LOGGER = StdoutLogger(
    correlator=ContextualCorrelator(),
    log_level=LogLevel.INFO,
    logger_id="parlant.bin.prepare_migration",
)


class VersionCheckpoint:
    def __init__(self, component: str, from_version: str, to_version: str):
        self.component = component
        self.from_version = from_version
        self.to_version = to_version

    def __str__(self) -> str:
        return f"{self.component}: {self.from_version} -> {self.to_version}"


MigrationFunction = Callable[[], Awaitable[None]]
migration_registry: dict[tuple[str, str, str], MigrationFunction] = {}


def register_migration(
    component: str, from_version: str, to_version: str
) -> Callable[[MigrationFunction], MigrationFunction]:
    """Decorator to register migration functions"""

    def decorator(func: MigrationFunction) -> MigrationFunction:
        migration_registry[(component, from_version, to_version)] = func
        return func

    return decorator


async def get_component_versions() -> list[tuple[str, str]]:
    """Get current versions of all components"""
    versions = []

    def _get_version_from_json_file(
        file_path: Path,
        collection_name: str,
    ) -> Optional[str]:
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            raw_data = json.load(f)
            if "metadata" in raw_data:
                return cast(str, raw_data["metadata"][0]["version"])
            else:
                items = raw_data.get(collection_name)
                if items and len(items) > 0:
                    return cast(str, items[0]["version"])
        return None

    agents_version = _get_version_from_json_file(
        PARLANT_HOME_DIR / "agents.json",
        "agents",
    )
    if agents_version:
        versions.append(("agents", agents_version))

    guidelines_version = _get_version_from_json_file(
        PARLANT_HOME_DIR / "guidelines.json",
        "guidelines",
    )
    if guidelines_version:
        versions.append(("guidelines", guidelines_version))

    context_vars_version = _get_version_from_json_file(
        PARLANT_HOME_DIR / "context_variables.json",
        "context_variables",
    )
    if context_vars_version:
        versions.append(("context_variables", context_vars_version))

    embedder_factory = EmbedderFactory(Container())
    glossary_db = await EXIT_STACK.enter_async_context(
        ChromaDatabase(LOGGER, PARLANT_HOME_DIR, embedder_factory)
    )
    with suppress(chromadb.errors.InvalidCollectionException):
        if glossary_db.chroma_client.get_collection("glossary_unembedded"):
            versions.append(
                ("glossary", cast(dict[str, Any], await glossary_db.read_metadata())["version"])
            )

    return versions


def backup_data() -> None:
    if Confirm.ask("Do you want to backup your data before migration?"):
        default_backup_dir = PARLANT_HOME_DIR.parent / "parlant-data.orig"
        try:
            backup_dir = Prompt.ask("Enter backup directory path", default=str(default_backup_dir))
            shutil.copytree(PARLANT_HOME_DIR, backup_dir, dirs_exist_ok=True)
            rich.print(f"[green]Data backed up to {backup_dir}")
        except Exception as e:
            rich.print(f"[red]Failed to backup data: {e}")
            die(f"Error backing up data: {e}")


async def create_metadata_collection(db: DocumentDatabase, collection_name: str) -> None:
    rich.print(f"[green]Migrating {collection_name} database...")
    try:
        collection = await db.get_collection(
            collection_name,
            BaseDocument,
            identity_loader,
        )

    except ValueError:
        rich.print(f"[yellow]Collection {collection_name} not found, skipping...")
        return

    try:
        metadata_collection = await db.get_collection(
            "metadata",
            BaseDocument,
            identity_loader,
        )
        await db.delete_collection("metadata")

    except ValueError:
        pass

    metadata_collection = await db.get_or_create_collection(
        "metadata",
        _MetadataDocument,
        identity_loader,
    )

    if document := await collection.find_one({}):
        await metadata_collection.insert_one(
            {
                "id": ObjectId(generate_id()),
                "version": document["version"],
            }
        )
        rich.print(f"[green]Successfully migrated {collection_name} database")
    else:
        rich.print(f"[yellow]No documents found in {collection_name} collection.")


async def migrate_glossary_with_metadata() -> None:
    rich.print("[green]Starting glossary migration...")
    try:
        embedder_factory = EmbedderFactory(Container())

        db = await EXIT_STACK.enter_async_context(
            ChromaDatabase(LOGGER, PARLANT_HOME_DIR, embedder_factory)
        )

        try:
            old_collection = db.chroma_client.get_collection("glossary")
        except chromadb.errors.InvalidCollectionException:
            rich.print("[yellow]Glossary collection not found, skipping...")
            return

        if docs := old_collection.peek(limit=1)["metadatas"]:
            document = docs[0]

            version = document["version"]

            embedder_module = importlib.import_module(
                f"{old_collection.metadata['embedder_module_path']}_service"
            )
            embedder_type = getattr(
                embedder_module,
                old_collection.metadata["embedder_type_path"],
            )

            all_items = old_collection.get(
                include=[IncludeEnum.documents, IncludeEnum.embeddings, IncludeEnum.metadatas]
            )
            rich.print(f"[green]Found {len(all_items['ids'])} items to migrate")

            chroma_unembedded_collection = next(
                (
                    collection
                    for collection in db.chroma_client.list_collections()
                    if collection.name == "glossary_unembedded"
                ),
                None,
            ) or db.chroma_client.create_collection(name="glossary_unembedded")

            chroma_new_collection = next(
                (
                    collection
                    for collection in db.chroma_client.list_collections()
                    if collection.name == db.format_collection_name("glossary", embedder_type)
                ),
                None,
            ) or db.chroma_client.create_collection(
                name=db.format_collection_name("glossary", embedder_type)
            )

            if all_items["metadatas"] is None:
                rich.print("[yellow]No metadatas found in glossary collection, skipping...")
                return

            for i in range(len(all_items["metadatas"])):
                assert all_items["documents"] is not None
                assert all_items["embeddings"] is not None

                new_doc = {
                    **all_items["metadatas"][i],
                    "checksum": md5_checksum(all_items["documents"][i]),
                }

                chroma_unembedded_collection.add(
                    ids=[all_items["ids"][i]],
                    documents=[str(new_doc["content"])],
                    metadatas=[cast(chromadb.types.Metadata, new_doc)],
                    embeddings=[0],
                )

                chroma_new_collection.add(
                    ids=[all_items["ids"][i]],
                    documents=[str(new_doc["content"])],
                    metadatas=[cast(chromadb.types.Metadata, new_doc)],
                    embeddings=all_items["embeddings"][i],
                )

            # Version starts at 1
            chroma_unembedded_collection.modify(
                metadata={"version": 1 + len(all_items["metadatas"])}
            )
            chroma_new_collection.modify(metadata={"version": 1 + len(all_items["metadatas"])})

            await db.upsert_metadata(
                "version",
                version,
            )
            rich.print("[green]Successfully migrated glossary data")

        db.chroma_client.delete_collection(old_collection.name)
        rich.print("[green]Cleaned up old glossary collection")

    except Exception as e:
        rich.print(f"[red]Failed to migrate glossary: {e}")
        die(f"Error migrating glossary: {e}")


@register_migration("agents", "0.1.0", "0.2.0")
async def migrate_agents_0_1_0_to_0_2_0() -> None:
    rich.print("[green]Starting migration for agents 0.1.0 -> 0.2.0")

    agents_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "agents.json")
    )
    await create_metadata_collection(agents_db, "agents")

    context_variables_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "context_variables.json")
    )
    await create_metadata_collection(context_variables_db, "variables")

    tags_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "tags.json")
    )
    await create_metadata_collection(tags_db, "tags")

    customers_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "customers.json")
    )
    await create_metadata_collection(customers_db, "customers")

    sessions_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "sessions.json")
    )
    await create_metadata_collection(sessions_db, "sessions")

    guideline_tool_associations_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "guideline_tool_associations.json")
    )
    await create_metadata_collection(guideline_tool_associations_db, "associations")

    guidelines_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "guidelines.json")
    )
    await create_metadata_collection(guidelines_db, "guidelines")

    guideline_connections_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "guideline_connections.json")
    )
    await create_metadata_collection(guideline_connections_db, "guideline_connections")

    evaluations_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "evaluations.json")
    )
    await create_metadata_collection(evaluations_db, "evaluations")

    services_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "services.json")
    )
    await create_metadata_collection(services_db, "tool_services")

    await migrate_glossary_with_metadata()

    await upgrade_document_database_metadata(agents_db, Version.String("0.2.0"))


@register_migration("guidelines", "0.1.0", "0.3.0")
async def migrate_guidelines_0_1_0_to_0_3_0() -> None:
    async def _association_document_loader(
        doc: BaseDocument,
    ) -> Optional[_GuidelineTagAssociationDocument]:
        return cast(_GuidelineTagAssociationDocument, doc)

    rich.print("[green]Starting migration for guidelines 0.1.0 -> 0.3.0")
    guidelines_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "guidelines.json")
    )

    guideline_collection = await guidelines_db.get_or_create_collection(
        "guidelines",
        BaseDocument,
        identity_loader,
    )

    guideline_tags_collection = await guidelines_db.get_or_create_collection(
        "guideline_tag_associations",
        _GuidelineTagAssociationDocument,
        _association_document_loader,
    )

    for guideline in await guideline_collection.find(filters={}):
        guideline_to_use = cast(_GuidelineDocument_v0_2_0, guideline)
        if guideline["version"] == "0.1.0":
            converted_guideline = await guideline_document_converter_0_1_0_to_0_2_0(guideline)
            if not converted_guideline:
                rich.print(f"[red]Failed to migrate guideline {guideline['id']}")
                continue
            guideline_to_use = cast(_GuidelineDocument_v0_2_0, converted_guideline)

        new_guideline = _GuidelineDocument(
            id=guideline_to_use["id"],
            version=Version.String("0.3.0"),
            creation_utc=guideline_to_use["creation_utc"],
            condition=guideline_to_use["condition"],
            action=guideline_to_use["action"],
            enabled=guideline_to_use["enabled"],
        )

        await guideline_collection.delete_one(
            filters={"id": {"$eq": ObjectId(guideline["id"])}},
        )

        await guideline_collection.insert_one(new_guideline)

        await guideline_tags_collection.insert_one(
            {
                "id": ObjectId(generate_id()),
                "version": Version.String("0.3.0"),
                "creation_utc": datetime.now(timezone.utc).isoformat(),
                "guideline_id": GuidelineId(guideline["id"]),
                "tag_id": Tag.for_agent_id(
                    cast(_GuidelineDocument_v0_1_0, guideline)["guideline_set"]
                ),
            }
        )

    await upgrade_document_database_metadata(guidelines_db, Version.String("0.3.0"))

    rich.print("[green]Successfully migrated guidelines to 0.3.0")


@register_migration("context_variables", "0.1.0", "0.2.0")
async def migrate_context_variables_0_1_0_to_0_2_0() -> None:
    async def _association_document_loader(
        doc: BaseDocument,
    ) -> Optional[_ContextVariableTagAssociationDocument]:
        return cast(_ContextVariableTagAssociationDocument, doc)

    context_variables_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "context_variables.json")
    )

    context_variables_collection = await context_variables_db.get_or_create_collection(
        "variables",
        BaseDocument,
        identity_loader,
    )
    context_variable_tags_collection = await context_variables_db.get_or_create_collection(
        "variable_tag_associations",
        _ContextVariableTagAssociationDocument,
        _association_document_loader,
    )

    for context_variable in await context_variables_collection.find(filters={}):
        await context_variable_tags_collection.insert_one(
            {
                "id": ObjectId(generate_id()),
                "version": Version.String("0.2.0"),
                "creation_utc": datetime.now(timezone.utc).isoformat(),
                "variable_id": ContextVariableId(context_variable["id"]),
                "tag_id": Tag.for_agent_id(
                    cast(_ContextVariableDocument_v0_1_0, context_variable)["variable_set"]
                ),
            }
        )

        await context_variables_collection.update_one(
            filters={"id": {"$eq": ObjectId(context_variable["id"])}},
            params={"version": Version.String("0.2.0")},
        )

    context_variable_values_collection = await context_variables_db.get_or_create_collection(
        "context_variable_values",
        BaseDocument,
        identity_loader,
    )

    for value in await context_variable_values_collection.find(filters={}):
        await context_variable_values_collection.update_one(
            filters={"id": {"$eq": ObjectId(value["id"])}},
            params={"version": Version.String("0.2.0")},
        )

    await upgrade_document_database_metadata(context_variables_db, Version.String("0.2.0"))

    rich.print("[green]Successfully migrated context variables to 0.2.0")


@register_migration("agents", "0.2.0", "0.3.0")
async def migrate_agents_0_2_0_to_0_3_0() -> None:
    agent_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "agents.json")
    )

    agent_collection = await agent_db.get_or_create_collection(
        "agents",
        BaseDocument,
        identity_loader,
    )

    await agent_db.get_or_create_collection(
        "agent_tags",
        BaseDocument,
        identity_loader,
    )

    for agent in await agent_collection.find(filters={}):
        if agent["version"] == "0.2.0":
            await agent_collection.update_one(
                filters={"id": {"$eq": ObjectId(agent["id"])}},
                params={"version": Version.String("0.3.0")},
            )

    await upgrade_document_database_metadata(agent_db, Version.String("0.3.0"))

    rich.print("[green]Successfully migrated agents from 0.2.0 to 0.3.0")


@register_migration("glossary", "0.1.0", "0.2.0")
async def migrate_glossary_0_1_0_to_0_2_0() -> None:
    rich.print("[green]Starting migration for glossary 0.1.0 -> 0.2.0")

    async def _association_document_loader(
        doc: BaseDocument,
    ) -> Optional[_TermTagAssociationDocument]:
        return cast(_TermTagAssociationDocument, doc)

    embedder_factory = EmbedderFactory(Container())

    db = await EXIT_STACK.enter_async_context(
        ChromaDatabase(LOGGER, PARLANT_HOME_DIR, embedder_factory)
    )

    glossary_tags_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "glossary_tags.json")
    )

    glossary_tags_collection = await glossary_tags_db.get_or_create_collection(
        "glossary_tags",
        _TermTagAssociationDocument,
        _association_document_loader,
    )

    chroma_unembedded_collection = next(
        (
            collection
            for collection in db.chroma_client.list_collections()
            if collection.name == "glossary_unembedded"
        ),
        None,
    ) or db.chroma_client.create_collection(name="glossary_unembedded")

    if metadatas := chroma_unembedded_collection.get()["metadatas"]:
        for doc in metadatas:
            new_doc = {
                "id": doc["id"],
                "version": Version.String("0.2.0"),
                "checksum": md5_checksum(
                    cast(str, doc["content"]) + datetime.now(timezone.utc).isoformat()
                ),
                "content": doc["content"],
                "creation_utc": doc["creation_utc"],
                "name": doc["name"],
                "description": doc["description"],
                "synonyms": doc["synonyms"],
            }

            chroma_unembedded_collection.delete(
                where=cast(chromadb.Where, {"id": {"$eq": cast(str, doc["id"])}})
            )
            chroma_unembedded_collection.add(
                ids=[cast(str, doc["id"])],
                documents=[cast(str, doc["content"])],
                metadatas=[cast(chromadb.Metadata, new_doc)],
                embeddings=[0],
            )

            await glossary_tags_collection.insert_one(
                {
                    "id": ObjectId(generate_id()),
                    "version": Version.String("0.2.0"),
                    "creation_utc": datetime.now(timezone.utc).isoformat(),
                    "term_id": TermId(cast(str, doc["id"])),
                    "tag_id": Tag.for_agent_id(cast(_TermDocument_v0_1_0, doc)["term_set"]),
                }
            )

    await db.upsert_metadata("version", Version.String("0.2.0"))

    rich.print("[green]Successfully migrated glossary from 0.1.0 to 0.2.0")


async def upgrade_document_database_metadata(
    db: DocumentDatabase,
    to_version: Version.String,
) -> None:
    metadata_collection = await db.get_or_create_collection(
        "metadata",
        BaseDocument,
        identity_loader,
    )

    if metadata_document := await metadata_collection.find_one(filters={}):
        await metadata_collection.update_one(
            filters={"id": {"$eq": metadata_document["id"]}},
            params={"version": to_version},
        )
    else:
        await metadata_collection.insert_one(
            {
                "id": ObjectId(generate_id()),
                "version": to_version,
            }
        )


async def detect_required_migrations() -> list[tuple[str, str, str]]:
    component_versions = await get_component_versions()
    required_migrations = []

    for component, current_version in component_versions:
        applicable_migrations = []
        for key in migration_registry:
            migration_component, from_version, to_version = key
            if migration_component == component:
                if current_version == from_version:
                    applicable_migrations.append(key)
                elif Version.from_string(current_version) > Version.from_string(
                    from_version
                ) and Version.from_string(current_version) < Version.from_string(to_version):
                    applicable_migrations.append(key)

        for migration in applicable_migrations:
            required_migrations.append(migration)

    return required_migrations


async def migrate() -> None:
    required_migrations = await detect_required_migrations()
    if not required_migrations:
        rich.print("[yellow]No migrations required.")
        return

    rich.print("[green]Starting migration process...")

    backup_data()

    applied_migrations = set()

    while required_migrations:
        for migration_key in required_migrations:
            if migration_key in applied_migrations:
                continue

            component, from_version, to_version = migration_key
            migration_func = migration_registry[migration_key]

            rich.print(f"[green]Running migration: {component} {from_version} -> {to_version}")
            await migration_func()
            applied_migrations.add(migration_key)

        new_required_migrations = await detect_required_migrations()
        required_migrations = [m for m in new_required_migrations if m not in applied_migrations]

        if not required_migrations:
            rich.print("[green]No more migrations required.")

    rich.print(
        f"[green]All migrations completed successfully. Applied {len(applied_migrations)} migrations in total."
    )


def die(message: str) -> NoReturn:
    rich.print(f"[red]{message}")
    print(message, file=sys.stderr)
    sys.exit(1)


def main() -> None:
    try:
        asyncio.run(migrate())
    except Exception as e:
        die(str(e))


if __name__ == "__main__":
    main()
