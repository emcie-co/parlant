# mypy: disable-error-code=import-untyped

import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass
from enum import Enum, auto
import os
from fastapi import FastAPI
from lagom import Container, Singleton
from typing import Any, AsyncIterator, Awaitable, Callable
import click
import click_completion
import json
from pathlib import Path
import sys
import uvicorn

from emcie.common.tools import ToolId
from emcie.server import VERSION
from emcie.server.adapters.db.chroma.glossary import GlossaryChromaStore
from emcie.server.adapters.nlp.openai import GPT_4o, GPT_4o_Mini, OpenAITextEmbedding3Large
from emcie.server.api.app import create_app
from emcie.server.core.contextual_correlator import ContextualCorrelator
from emcie.server.core.agents import AgentDocumentStore, AgentStore
from emcie.server.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
from emcie.server.core.emission.event_publisher import EventPublisherFactory
from emcie.server.core.emissions import EventEmitterFactory
from emcie.server.core.end_users import EndUserDocumentStore, EndUserStore
from emcie.server.core.evaluations import EvaluationDocumentStore, EvaluationStatus, EvaluationStore
from emcie.server.core.guideline_connections import (
    GuidelineConnectionDocumentStore,
    GuidelineConnectionStore,
)
from emcie.server.core.guidelines import (
    GuidelineDocumentStore,
    GuidelineStore,
)
from emcie.server.adapters.db.chroma.database import ChromaDatabase
from emcie.server.adapters.db.json_file import JSONFileDocumentDatabase
from emcie.server.core.nlp.embedding import EmbedderFactory
from emcie.server.core.nlp.generation import SchematicGenerator
from emcie.server.core.services.tools.service_registry import (
    ServiceRegistry,
    ServiceDocumentRegistry,
)
from emcie.server.core.sessions import (
    PollingSessionListener,
    SessionDocumentStore,
    SessionListener,
    SessionStore,
)
from emcie.server.core.glossary import GlossaryStore
from emcie.server.core.tools import LocalToolService, ToolService, MultiplexedToolService
from emcie.server.core.services.tools.plugins import PluginClient
from emcie.server.core.engines.alpha.engine import AlphaEngine
from emcie.server.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
    GuidelineToolAssociationStore,
)
from emcie.server.core.engines.alpha.tool_caller import ToolCallInferenceSchema
from emcie.server.core.engines.alpha.guideline_proposer import (
    GuidelineProposer,
    GuidelinePropositionsSchema,
)
from emcie.server.core.engines.alpha.message_event_producer import (
    MessageEventProducer,
    MessageEventSchema,
)
from emcie.server.core.engines.alpha.tool_event_producer import ToolEventProducer
from emcie.server.core.engines.types import Engine
from emcie.server.core.services.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
)
from emcie.server.core.services.indexing.coherence_checker import (
    CoherenceChecker,
    PredicatesEntailmentTestsSchema,
    ActionsContradictionTestsSchema,
)
from emcie.server.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
    GuidelineConnectionPropositionsSchema,
)
from emcie.server.core.services.indexing.indexer import Indexer
from emcie.server.core.logging import FileLogger, Logger
from emcie.server.core.mc import MC
from emcie.server.configuration_validator import ConfigurationFileValidator

DEFAULT_PORT = 8000
SERVER_ADDRESS = "https://localhost"

EMCIE_HOME_DIR = Path(os.environ.get("EMCIE_HOME", "/var/lib/emcie"))
EMCIE_HOME_DIR.mkdir(parents=True, exist_ok=True)

MULTIPLEXED_TOOL_SERVICE = MultiplexedToolService()
TOOL_NAME_TO_ID: dict[str, ToolId] = {}

EXIT_STACK: AsyncExitStack

sys.path.append(EMCIE_HOME_DIR.as_posix())


CORRELATOR = ContextualCorrelator()
LOGGER = FileLogger(EMCIE_HOME_DIR / "emcie.log", CORRELATOR)

LOGGER.info(f"Emcie server version {VERSION}")
LOGGER.info(f"Using home directory '{EMCIE_HOME_DIR.absolute()}'")


class StartupError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class CLIParams:
    config_file: Path
    config: dict[str, Any]
    port: int
    index: bool
    force: bool


class ShutdownReason(Enum):
    HOT_RELOAD = auto()
    SHUTDOWN_REQUEST = auto()


async def load_agents(c: Container, config: Any) -> None:
    store = c[AgentStore]
    existing_agents = await store.list_agents()

    for agent in config["agents"]:
        if not [a for a in existing_agents if a.name == agent["name"]]:
            await store.create_agent(
                name=agent["name"],
                description=agent.get("description"),
            )


async def load_tools(c: Container, config: Any) -> None:
    local_tool_service = c[LocalToolService]

    for service in config["services"]:
        if service["type"] == "local":
            for tool_name, tool_entry in service["tools"].items():
                tool = await local_tool_service.create_tool(
                    name=tool_entry["function_name"],
                    module_path=tool_entry["module_path"],
                    description=tool_entry["description"],
                    parameters=tool_entry["parameters"],
                    required=tool_entry["required"],
                    consequential=False,
                )

                TOOL_NAME_TO_ID[tool_name] = tool.id
        elif service["type"] == "plugin":
            name = str(service["name"])
            url = str(service["url"])

            client = PluginClient(
                url=url,
                event_emitter_factory=c[EventEmitterFactory],
                correlator=CORRELATOR,
            )

            await EXIT_STACK.enter_async_context(client)

            MULTIPLEXED_TOOL_SERVICE.add_service(name, client)


async def load_guidelines(c: Container, config: Any) -> None:
    agent_store = c[AgentStore]
    guideline_store = c[GuidelineStore]
    guideline_tool_association_store = c[GuidelineToolAssociationStore]

    agents = await agent_store.list_agents()

    for agent_name, guidelines in config["guidelines"].items():
        agent_id = next(a.id for a in agents if a.name == agent_name)

        for guideline_spec in guidelines:
            guideline = await guideline_store.create_guideline(
                guideline_set=agent_id,
                predicate=guideline_spec["when"],
                action=guideline_spec["then"],
            )

            for tool_name in guideline_spec.get("enabled_tools", []):
                await guideline_tool_association_store.create_association(
                    guideline_id=guideline.id,
                    tool_id=TOOL_NAME_TO_ID.get(tool_name, ToolId(tool_name)),
                )


@asynccontextmanager
async def setup_container(config: Any) -> AsyncIterator[Container]:
    TOOL_NAME_TO_ID.clear()
    MULTIPLEXED_TOOL_SERVICE.services.clear()

    for store_name in [
        "guidelines",
        "tools",
        "guideline_tool_associations",
        "context_variables",
    ]:
        (EMCIE_HOME_DIR / f"{store_name}.json").unlink(missing_ok=True)

    c = Container()

    c[ContextualCorrelator] = CORRELATOR
    c[Logger] = LOGGER

    c[SchematicGenerator[GuidelinePropositionsSchema]] = GPT_4o[GuidelinePropositionsSchema](
        logger=LOGGER
    )
    c[SchematicGenerator[MessageEventSchema]] = GPT_4o[MessageEventSchema](logger=LOGGER)
    c[SchematicGenerator[ToolCallInferenceSchema]] = GPT_4o_Mini[ToolCallInferenceSchema](
        logger=LOGGER
    )
    c[SchematicGenerator[PredicatesEntailmentTestsSchema]] = GPT_4o[
        PredicatesEntailmentTestsSchema
    ](logger=LOGGER)
    c[SchematicGenerator[ActionsContradictionTestsSchema]] = GPT_4o[
        ActionsContradictionTestsSchema
    ](logger=LOGGER)
    c[SchematicGenerator[GuidelineConnectionPropositionsSchema]] = GPT_4o[
        GuidelineConnectionPropositionsSchema
    ](logger=LOGGER)

    agents_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, EMCIE_HOME_DIR / "agents.json")
    )
    context_variables_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, EMCIE_HOME_DIR / "context_variables.json")
    )
    end_users_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, EMCIE_HOME_DIR / "end_users.json")
    )
    sessions_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(
            LOGGER,
            EMCIE_HOME_DIR / "sessions.json",
        )
    )
    guidelines_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, EMCIE_HOME_DIR / "guidelines.json")
    )
    tools_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, EMCIE_HOME_DIR / "tools.json")
    )
    guideline_tool_associations_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, EMCIE_HOME_DIR / "guideline_tool_associations.json")
    )
    guideline_connections_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, EMCIE_HOME_DIR / "guideline_connections.json")
    )
    evaluations_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, EMCIE_HOME_DIR / "evaluations.json")
    )
    services_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, EMCIE_HOME_DIR / "services.json")
    )

    c[AgentStore] = AgentDocumentStore(agents_db)
    c[ContextVariableStore] = ContextVariableDocumentStore(context_variables_db)
    c[EndUserStore] = EndUserDocumentStore(end_users_db)
    c[GuidelineStore] = GuidelineDocumentStore(guidelines_db)

    c[LocalToolService] = LocalToolService(tools_db)
    MULTIPLEXED_TOOL_SERVICE.services["local"] = c[LocalToolService]
    c[ToolService] = MULTIPLEXED_TOOL_SERVICE

    c[GuidelineToolAssociationStore] = GuidelineToolAssociationDocumentStore(
        guideline_tool_associations_db
    )
    c[GuidelineConnectionStore] = GuidelineConnectionDocumentStore(guideline_connections_db)
    c[SessionStore] = SessionDocumentStore(sessions_db)
    c[SessionListener] = PollingSessionListener
    c[GlossaryStore] = GlossaryChromaStore(
        ChromaDatabase(LOGGER, EMCIE_HOME_DIR, EmbedderFactory(c)),
        embedder_type=OpenAITextEmbedding3Large,
    )

    c[EvaluationStore] = EvaluationDocumentStore(evaluations_db)

    c[GuidelineProposer] = GuidelineProposer(
        c[Logger],
        c[SchematicGenerator[GuidelinePropositionsSchema]],
    )
    c[GuidelineConnectionProposer] = GuidelineConnectionProposer(
        c[Logger],
        c[SchematicGenerator[GuidelineConnectionPropositionsSchema]],
        c[GlossaryStore],
    )
    c[ToolEventProducer] = ToolEventProducer(
        c[Logger],
        c[ContextualCorrelator],
        c[ToolService],
        c[SchematicGenerator[ToolCallInferenceSchema]],
    )
    c[MessageEventProducer] = MessageEventProducer(
        c[Logger],
        c[ContextualCorrelator],
        c[SchematicGenerator[MessageEventSchema]],
    )

    c[CoherenceChecker] = CoherenceChecker(
        c[Logger],
        c[SchematicGenerator[PredicatesEntailmentTestsSchema]],
        c[SchematicGenerator[ActionsContradictionTestsSchema]],
        c[GlossaryStore],
    )

    c[BehavioralChangeEvaluator] = BehavioralChangeEvaluator(
        c[Logger],
        c[AgentStore],
        c[EvaluationStore],
        c[GuidelineStore],
        c[GuidelineConnectionProposer],
        c[CoherenceChecker],
    )

    c[EventEmitterFactory] = Singleton(EventPublisherFactory)

    c[ServiceRegistry] = await EXIT_STACK.enter_async_context(
        ServiceDocumentRegistry(
            services_db,
            c[EventEmitterFactory],
            c[ContextualCorrelator],
        )
    )

    c[Engine] = AlphaEngine

    for loader in load_agents, load_tools, load_guidelines:
        await loader(c, config)
    c[MC] = await EXIT_STACK.enter_async_context(MC(c))
    yield c


async def recover_server_tasks(
    evaluation_store: EvaluationStore,
    evaluator: BehavioralChangeEvaluator,
) -> None:
    for evaluation in await evaluation_store.list_evaluations():
        if evaluation.status in [EvaluationStatus.PENDING, EvaluationStatus.RUNNING]:
            await evaluator.run_evaluation(evaluation)


@asynccontextmanager
async def load_app(params: CLIParams) -> AsyncIterator[FastAPI]:
    global EXIT_STACK

    EXIT_STACK = AsyncExitStack()

    async with setup_container(params.config) as container, EXIT_STACK:
        indexer = Indexer(
            index_file=EMCIE_HOME_DIR / "index.json",
            logger=container[Logger],
            guideline_store=container[GuidelineStore],
            guideline_connection_store=container[GuidelineConnectionStore],
            agent_store=container[AgentStore],
            guideline_connection_proposer=container[GuidelineConnectionProposer],
        )

        if not params.index:
            if params.force:
                LOGGER.warning("Skipping indexing. This might cause unpredictable behavior.")

            elif await indexer.should_index():
                raise StartupError("indexing needs to be perform.")
        else:
            await indexer.index()

        await recover_server_tasks(
            evaluation_store=container[EvaluationStore],
            evaluator=container[BehavioralChangeEvaluator],
        )

        yield await create_app(container)


async def serve_app(
    app: FastAPI,
    port: int,
    should_hot_reload: Callable[[], Awaitable[bool]],
) -> ShutdownReason:
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)
    interrupted = False

    async def monitor_shutdown_request() -> ShutdownReason:
        try:
            while True:
                await asyncio.sleep(1)

                if await should_hot_reload():
                    server.should_exit = True
                    return ShutdownReason.HOT_RELOAD
                elif interrupted:
                    return ShutdownReason.SHUTDOWN_REQUEST
        except asyncio.CancelledError:
            return ShutdownReason.SHUTDOWN_REQUEST

    shutdown_monitor_task = asyncio.create_task(monitor_shutdown_request())

    try:
        await server.serve()
        interrupted = True
    except (KeyboardInterrupt, asyncio.CancelledError):
        return ShutdownReason.SHUTDOWN_REQUEST
    except BaseException as e:
        LOGGER.critical(e.__class__.__name__ + ": " + str(e))
        sys.exit(1)

    return await shutdown_monitor_task


async def start_server(params: CLIParams) -> None:
    assert ConfigurationFileValidator(LOGGER).validate(config_file=params.config_file)

    last_config_update_time = params.config_file.stat().st_mtime

    async def config_file_changed() -> bool:
        nonlocal last_config_update_time
        current_mtime = params.config_file.stat().st_mtime

        if current_mtime > last_config_update_time:
            validated = ConfigurationFileValidator(LOGGER).validate(config_file=params.config_file)

            last_config_update_time = current_mtime

            return validated

        return False

    while True:
        async with load_app(params) as app:
            shutdown_reason = await serve_app(
                app,
                params.port,
                should_hot_reload=config_file_changed,
            )

            if shutdown_reason == ShutdownReason.SHUTDOWN_REQUEST:
                return
            elif shutdown_reason == ShutdownReason.HOT_RELOAD:
                LOGGER.info("***** HOT RELOAD *****")

                with open(params.config_file) as f:
                    params.config = json.load(f)
                    last_config_update_time = params.config_file.stat().st_mtime


def main() -> None:
    click_completion.init()

    @click.group
    @click.option(
        "-c",
        "--config-file",
        type=str,
        help="Server configuration file",
        metavar="FILE",
        required=True,
        default=EMCIE_HOME_DIR / "config.json",
    )
    @click.pass_context
    def cli(ctx: click.Context, config_file: str) -> None:
        if not ctx.obj:
            config_file_path = Path(config_file)

            if not config_file_path.exists():
                print(f"error: config file not found: {config_file_path}", file=sys.stderr)
                sys.exit(1)

            with open(config_file_path, "r") as f:
                config = json.load(f)

                ctx.obj = CLIParams(
                    config_file=config_file_path,
                    config=config,
                    port=DEFAULT_PORT,
                    index=False,
                    force=True,
                )

    @cli.command(help="Run the Emcie server")
    @click.option(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port",
    )
    @click.option(
        "--index/--no-index",
        type=bool,
        show_default=True,
        default=False,
        help="Index configuration changes on startup",
    )
    @click.option(
        "-f",
        "--force",
        type=bool,
        default=False,
        is_flag=True,
        help="Ignore warnings and checks",
    )
    @click.pass_context
    def run(ctx: click.Context, port: int, index: bool, force: bool) -> None:
        ctx.obj.port = port
        ctx.obj.index = index
        ctx.obj.force = force
        asyncio.run(start_server(ctx.obj))

    try:
        cli()
    except StartupError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
