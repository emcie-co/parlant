import pytest
from parlant.core.context_variables import ContextVariableDocumentStore
from parlant.core.common import IdGenerator
from parlant.core.persistence.document_database import DocumentDatabase

@pytest.fixture
def id_generator() -> IdGenerator:
    return IdGenerator()

@pytest.fixture
async def database() -> DocumentDatabase:
    return await DocumentDatabase.create_memory_instance()

@pytest.fixture
async def store(id_generator: IdGenerator, database: DocumentDatabase) -> ContextVariableDocumentStore:
    return await ContextVariableDocumentStore(
        id_generator=id_generator, database=database, allow_migration=True
    ).__aenter__()

@pytest.mark.asyncio
async def test_create_variable_with_invalid_freshness_rules_raises_value_error(
    store: ContextVariableDocumentStore,
):
    with pytest.raises(ValueError):
        await store.create_variable(
            name="test",
            freshness_rules="invalid cron expression",
        )

@pytest.mark.asyncio
async def test_update_variable_with_invalid_freshness_rules_raises_value_error(
    store: ContextVariableDocumentStore,
):
    variable = await store.create_variable(name="test")

    with pytest.raises(ValueError):
        await store.update_variable(
            id=variable.id,
            params={"freshness_rules": "invalid cron expression"},
        )
