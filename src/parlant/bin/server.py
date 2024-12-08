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

# mypy: disable-error-code=import-untyped

import asyncio
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass
import os
from lagom import Container, Singleton
from typing import AsyncIterator, Callable
from typing_extensions import NoReturn
import click
import click_completion
from pathlib import Path
import sys
import uvicorn

from parlant.core.nlp.service import NLPService
from parlant.core.tags import TagDocumentStore, TagStore
from parlant.version import VERSION
from parlant.adapters.db.chroma.glossary import GlossaryChromaStore
from parlant.api.app import create_api_app, ASGIApplication
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.agents import AgentDocumentStore, AgentStore
from parlant.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
from parlant.core.emission.event_publisher import EventPublisherFactory
from parlant.core.emissions import EventEmitterFactory
from parlant.core.customers import CustomerDocumentStore, CustomerStore
from parlant.core.evaluations import (
    EvaluationListener,
    PollingEvaluationListener,
    EvaluationDocumentStore,
    EvaluationStatus,
    EvaluationStore,
)
from parlant.core.guideline_connections import (
    GuidelineConnectionDocumentStore,
    GuidelineConnectionStore,
)
from parlant.core.guidelines import (
    GuidelineDocumentStore,
    GuidelineStore,
)
from parlant.adapters.db.chroma.database import ChromaDatabase
from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.core.nlp.embedding import EmbedderFactory
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.services.tools.service_registry import (
    ServiceRegistry,
    ServiceDocumentRegistry,
)
from parlant.core.sessions import (
    PollingSessionListener,
    SessionDocumentStore,
    SessionListener,
    SessionStore,
)
from parlant.core.glossary import GlossaryStore
from parlant.core.engines.alpha.engine import AlphaEngine
from parlant.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
    GuidelineToolAssociationStore,
)
from parlant.core.engines.alpha.tool_caller import ToolCallInferenceSchema
from parlant.core.engines.alpha.guideline_proposer import (
    GuidelineProposer,
    GuidelinePropositionsSchema,
)
from parlant.core.engines.alpha.message_event_generator import (
    MessageEventGenerator,
    MessageEventSchema,
)
from parlant.core.engines.alpha.tool_event_generator import ToolEventGenerator
from parlant.core.engines.types import Engine
from parlant.core.services.indexing.behavioral_change_evaluation import (
    BehavioralChangeEvaluator,
)
from parlant.core.services.indexing.coherence_checker import (
    CoherenceChecker,
    ConditionsEntailmentTestsSchema,
    ActionsContradictionTestsSchema,
)
from parlant.core.services.indexing.guideline_connection_proposer import (
    GuidelineConnectionProposer,
    GuidelineConnectionPropositionsSchema,
)
from parlant.core.logging import FileLogger, LogLevel, Logger
from parlant.core.application import Application

DEFAULT_PORT = 8800
SERVER_ADDRESS = "https://localhost"

DEFAULT_NLP_SERVICE = "openai"

PARLANT_HOME_DIR = Path(os.environ.get("PARLANT_HOME", "runtime-data"))
PARLANT_HOME_DIR.mkdir(parents=True, exist_ok=True)

EXIT_STACK: AsyncExitStack

DEFAULT_AGENT_NAME = "Default Agent"

sys.path.append(PARLANT_HOME_DIR.as_posix())


CORRELATOR = ContextualCorrelator()
LOGGER = FileLogger(PARLANT_HOME_DIR / "parlant.log", CORRELATOR, LogLevel.INFO)
BACKGROUND_TASK_SERVICE = BackgroundTaskService(LOGGER)

LOGGER.info(f"Parlant server version {VERSION}")
LOGGER.info(f"Using home directory '{PARLANT_HOME_DIR.absolute()}'")


class StartupError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class CLIParams:
    port: int
    nlp_service: str
    log_level: str


def load_anthropic() -> NLPService:
    from parlant.adapters.nlp.anthropic import AnthropicService

    return AnthropicService(LOGGER)


def load_aws() -> NLPService:
    from parlant.adapters.nlp.aws import BedrockService

    return BedrockService(LOGGER)


def load_azure() -> NLPService:
    from parlant.adapters.nlp.azure import AzureService

    return AzureService(LOGGER)


def load_cerebras() -> NLPService:
    from parlant.adapters.nlp.cerebras import CerebrasService

    return CerebrasService(LOGGER)


def load_gemini() -> NLPService:
    from parlant.adapters.nlp.gemini import GeminiService

    return GeminiService(LOGGER)


def load_openai() -> NLPService:
    from parlant.adapters.nlp.openai import OpenAIService

    return OpenAIService(LOGGER)


def load_together() -> NLPService:
    from parlant.adapters.nlp.together import TogetherService

    return TogetherService(LOGGER)


async def create_agent_if_absent(agent_store: AgentStore) -> None:
    agents = await agent_store.list_agents()
    if not agents:
        await agent_store.create_agent(name=DEFAULT_AGENT_NAME)


@asynccontextmanager
async def setup_container(nlp_service_name: str) -> AsyncIterator[Container]:
    c = Container()

    c[ContextualCorrelator] = CORRELATOR
    c[Logger] = LOGGER

    agents_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "agents.json")
    )
    context_variables_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "context_variables.json")
    )
    tags_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "tags.json")
    )
    customers_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "customers.json")
    )
    sessions_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(
            LOGGER,
            PARLANT_HOME_DIR / "sessions.json",
        )
    )
    guidelines_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "guidelines.json")
    )
    guideline_tool_associations_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "guideline_tool_associations.json")
    )
    guideline_connections_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "guideline_connections.json")
    )
    evaluations_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "evaluations.json")
    )
    services_db = await EXIT_STACK.enter_async_context(
        JSONFileDocumentDatabase(LOGGER, PARLANT_HOME_DIR / "services.json")
    )

    c[AgentStore] = await EXIT_STACK.enter_async_context(AgentDocumentStore(agents_db))
    c[ContextVariableStore] = await EXIT_STACK.enter_async_context(
        ContextVariableDocumentStore(context_variables_db)
    )
    c[TagStore] = await EXIT_STACK.enter_async_context(TagDocumentStore(tags_db))
    c[CustomerStore] = await EXIT_STACK.enter_async_context(CustomerDocumentStore(customers_db))
    c[GuidelineStore] = await EXIT_STACK.enter_async_context(GuidelineDocumentStore(guidelines_db))
    c[GuidelineToolAssociationStore] = await EXIT_STACK.enter_async_context(
        GuidelineToolAssociationDocumentStore(guideline_tool_associations_db)
    )
    c[GuidelineConnectionStore] = await EXIT_STACK.enter_async_context(
        GuidelineConnectionDocumentStore(guideline_connections_db)
    )
    c[SessionStore] = await EXIT_STACK.enter_async_context(SessionDocumentStore(sessions_db))
    c[SessionListener] = PollingSessionListener

    c[EvaluationStore] = await EXIT_STACK.enter_async_context(
        EvaluationDocumentStore(evaluations_db)
    )
    c[EvaluationListener] = PollingEvaluationListener

    c[EventEmitterFactory] = Singleton(EventPublisherFactory)

    c[BackgroundTaskService] = await EXIT_STACK.enter_async_context(BACKGROUND_TASK_SERVICE)

    nlp_service_initializer: dict[str, Callable[[], NLPService]] = {
        "anthropic": load_anthropic,
        "aws": load_aws,
        "azure": load_azure,
        "cerebras": load_cerebras,
        "gemini": load_gemini,
        "openai": load_openai,
        "together": load_together,
    }

    c[ServiceRegistry] = await EXIT_STACK.enter_async_context(
        ServiceDocumentRegistry(
            database=services_db,
            event_emitter_factory=c[EventEmitterFactory],
            correlator=c[ContextualCorrelator],
            nlp_services={nlp_service_name: nlp_service_initializer[nlp_service_name]()},
        )
    )

    nlp_service = await c[ServiceRegistry].read_nlp_service(nlp_service_name)

    c[NLPService] = nlp_service

    c[GlossaryStore] = await EXIT_STACK.enter_async_context(
        GlossaryChromaStore(
            await EXIT_STACK.enter_async_context(
                ChromaDatabase(LOGGER, PARLANT_HOME_DIR, EmbedderFactory(c)),
            ),
            embedder_type=type(await nlp_service.get_embedder()),
        )
    )

    c[SchematicGenerator[GuidelinePropositionsSchema]] = await nlp_service.get_schematic_generator(
        GuidelinePropositionsSchema
    )
    c[SchematicGenerator[MessageEventSchema]] = await nlp_service.get_schematic_generator(
        MessageEventSchema
    )
    c[SchematicGenerator[ToolCallInferenceSchema]] = await nlp_service.get_schematic_generator(
        ToolCallInferenceSchema
    )
    c[
        SchematicGenerator[ConditionsEntailmentTestsSchema]
    ] = await nlp_service.get_schematic_generator(ConditionsEntailmentTestsSchema)
    c[
        SchematicGenerator[ActionsContradictionTestsSchema]
    ] = await nlp_service.get_schematic_generator(ActionsContradictionTestsSchema)
    c[
        SchematicGenerator[GuidelineConnectionPropositionsSchema]
    ] = await nlp_service.get_schematic_generator(GuidelineConnectionPropositionsSchema)

    c[GuidelineProposer] = GuidelineProposer(
        c[Logger],
        c[SchematicGenerator[GuidelinePropositionsSchema]],
    )
    c[GuidelineConnectionProposer] = GuidelineConnectionProposer(
        c[Logger],
        c[SchematicGenerator[GuidelineConnectionPropositionsSchema]],
        c[GlossaryStore],
    )

    c[CoherenceChecker] = CoherenceChecker(
        c[Logger],
        c[SchematicGenerator[ConditionsEntailmentTestsSchema]],
        c[SchematicGenerator[ActionsContradictionTestsSchema]],
        c[GlossaryStore],
    )

    c[BehavioralChangeEvaluator] = BehavioralChangeEvaluator(
        c[Logger],
        c[BackgroundTaskService],
        c[AgentStore],
        c[EvaluationStore],
        c[GuidelineStore],
        c[GuidelineConnectionProposer],
        c[CoherenceChecker],
    )

    c[MessageEventGenerator] = MessageEventGenerator(
        c[Logger],
        c[ContextualCorrelator],
        c[SchematicGenerator[MessageEventSchema]],
    )

    c[ToolEventGenerator] = ToolEventGenerator(
        c[Logger],
        c[ContextualCorrelator],
        c[ServiceRegistry],
        c[SchematicGenerator[ToolCallInferenceSchema]],
    )

    c[Engine] = AlphaEngine

    c[Application] = Application(c)

    yield c


async def recover_server_tasks(
    evaluation_store: EvaluationStore,
    evaluator: BehavioralChangeEvaluator,
) -> None:
    for evaluation in await evaluation_store.list_evaluations():
        if evaluation.status in [EvaluationStatus.PENDING, EvaluationStatus.RUNNING]:
            LOGGER.info(f"Recovering evaluation task: '{evaluation.id}'")
            await evaluator.run_evaluation(evaluation)


@asynccontextmanager
async def load_app(params: CLIParams) -> AsyncIterator[ASGIApplication]:
    global EXIT_STACK

    EXIT_STACK = AsyncExitStack()

    async with setup_container(params.nlp_service) as container, EXIT_STACK:
        await recover_server_tasks(
            evaluation_store=container[EvaluationStore],
            evaluator=container[BehavioralChangeEvaluator],
        )

        await create_agent_if_absent(container[AgentStore])

        yield await create_api_app(container)


async def serve_app(
    app: ASGIApplication,
    port: int,
) -> None:
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
        await asyncio.sleep(0)  # Required to trigger the possible cancellation error
    except (KeyboardInterrupt, asyncio.CancelledError):
        await BACKGROUND_TASK_SERVICE.cancel_all(reason="Server shutting down")
    except BaseException as e:
        LOGGER.critical(e.__class__.__name__ + ": " + str(e))
        sys.exit(1)


def die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    sys.exit(1)


def require_env_keys(keys: list[str]) -> None:
    if missing_keys := [k for k in keys if not os.environ.get(k)]:
        die(f"The following environment variables are missing:\n{', '.join(missing_keys)}")


async def start_server(params: CLIParams) -> None:
    LOGGER.set_level(
        {
            "info": LogLevel.INFO,
            "debug": LogLevel.DEBUG,
            "warning": LogLevel.WARNING,
            "error": LogLevel.ERROR,
            "critical": LogLevel.CRITICAL,
        }[params.log_level],
    )

    async with load_app(params) as app:
        await serve_app(
            app,
            params.port,
        )


def main() -> None:
    click_completion.init()

    @click.group(invoke_without_command=True)
    @click.option(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Server port",
    )
    @click.option(
        "--openai",
        is_flag=True,
        help="Run with OpenAI. The environment variable OPENAI_API_KEY must be set",
        default=True,
    )
    @click.option(
        "--aws",
        is_flag=True,
        help="Run with AWS Bedrock. The following environment variables must be set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION",
        default=False,
    )
    @click.option(
        "--azure",
        is_flag=True,
        help="Run with Azure OpenAI. The following environment variables must be set: AZURE_API_KEY, AZURE_ENDPOINT",
        default=False,
    )
    @click.option(
        "--gemini",
        is_flag=True,
        help="Run with Gemini. The environment variable GEMINI_API_KEY must be set",
        default=False,
    )
    @click.option(
        "--anthropic",
        is_flag=True,
        help="Run with Anthropic. The environment variable ANTHROPIC_API_KEY must be set",
        default=False,
    )
    @click.option(
        "--cerebras",
        is_flag=True,
        help="Run with Cerebras. The environment variable CEREBRAS_API_KEY must be set",
        default=False,
    )
    @click.option(
        "--together",
        is_flag=True,
        help="Run with Together AI. The environment variable TOGETHER_API_KEY must be set",
        default=False,
    )
    @click.option(
        "--log-level",
        type=click.Choice(["debug", "info", "warning", "error", "critical"]),
        default="info",
        help="Log level",
    )
    @click.pass_context
    def cli(
        ctx: click.Context,
        port: int,
        openai: bool,
        aws: bool,
        azure: bool,
        gemini: bool,
        anthropic: bool,
        cerebras: bool,
        together: bool,
        log_level: str,
    ) -> None:
        if sum([openai, aws, azure, gemini, anthropic, cerebras, together]) > 2:
            print("error: only one NLP service profile can be selected")
            sys.exit(1)

        non_default_service_selected = any((aws, azure, gemini, anthropic, cerebras, together))

        if not non_default_service_selected:
            nlp_service = "openai"
            require_env_keys(["OPENAI_API_KEY"])
        elif aws:
            nlp_service = "aws"
            require_env_keys(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"])
        elif azure:
            nlp_service = "azure"
            require_env_keys(["AZURE_API_KEY", "AZURE_ENDPOINT"])
        elif gemini:
            nlp_service = "gemini"
            require_env_keys(["GEMINI_API_KEY"])
        elif anthropic:
            nlp_service = "anthropic"
            require_env_keys(["ANTHROPIC_API_KEY"])
        elif cerebras:
            nlp_service = "cerebras"
            require_env_keys(["CEREBRAS_API_KEY"])
        elif together:
            nlp_service = "together"
            require_env_keys(["TOGETHER_API_KEY"])
        else:
            assert False, "Should never get here"

        ctx.obj = CLIParams(
            port=port,
            nlp_service=nlp_service,
            log_level=log_level,
        )

        asyncio.run(start_server(ctx.obj))

    try:
        cli()
    except StartupError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
