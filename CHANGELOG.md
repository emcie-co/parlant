# Changelog

All notable changes to Parlant will be documented here.

## [Unreleased]

TBD

## [3.1.1] - 2026-01-05

### Changed

- Set OpenAI as default service until EmcieService is live

## [3.1.0] - 2026-01-05

### Added

- Add .current property for Server, Agent, and Customer in SDK
- Add /healthz endpoint
- Add API for CRUD operations on session metadata
- Add EmcieService
- Add GLM service
- Add Mistral service
- Add OpenRouter service
- Add OpenTelemetry integration for Meter, Logger and Tracer
- Add Qdrant VectorDatabase adapter
- Add Snowflake Cortex service
- Add ability to configure and extend the FastAPI app object
- Add deferred retrievers
- Add dynamic composition mode
- Add follow-up canned responses
- Add guideline criticality level
- Add guideline on_match() hooks
- Add persistence option for context variable values (variable store)
- Added guideline descriptions
- Allow bailing out of canned response selection and utilize the draft directly, using a hook
- Allow controlling max tool result payload via environment variable
- Allow controlling perceived performance policy per agent
- Allow journey transitions from one tool state to another
- Allow specifying custom IDs when creating agents via SDK and API
- Allow specifying custom IDs when creating customers via SDK and API
- Allow specifying custom IDs when creating guidelines, journeys, and glossary terms via SDK and API
- Expose IoC container in server object
- Support adding custom canrep fields to matched guidelines and journey states
- Support code-based, custom guideline matchers

### Changed

- Changed default NLPService to EmcieService
- Improved efficiency of journey state matching when first state is a tool state
- Rename ContextualCorrelator to Tracer
- Rename LoadedContext to EngineContext
- Support proxy URL for LiteLLM

### Fixed

- Fix critical bug with cancellation during response analysis
- Fix critical similarity calculation error in TransientVectorDatabase
- Fix unnecessary extra evaluation of journeys and tools in some edge cases
- Improved Gemini Flash 2.5 output consistency by using function call trick instead of structured outputs

## [3.0.4] - 2025-11-18

### Fixed

- Fix bug where NanoDB query failed when no filters matched
- Extend tool insights across iterations
- Fix deprecated status.HTTP_422_UNPROCESSABLE_ENTITY to status.HTTP_422_UNPROCESSABLE_CONTENT
- Fix broken CLI by adding missing websocket-client dependency
- Added specific classes for embedder initialisation
- Make base url once in OllamaEmbedder
- Update dependencies for security, upgrade FastAPI, fix mypy in hugging_face.py
- Bump torch for fixing vulnerability

## [3.0.3] - 2025-10-23

### Fixed

- Fix installation issue in some environments, failing due to an older FastMCP version
- Bump versions of OpenTelemetry
- Made ChromaDB an extra package parlant[chroma]
- Update NPM dependencies for integrated UI

## [3.0.2] - 2025-08-27

### Added

- Added docs/\* and llms.txt
- Added Vertex NLP service
- Added Ollama NLP service
- Added LiteLLM support to the SDK
- Added Gemini support to the SDK
- Added Journey.create_observation() helper
- Added auth permission READ_AGENT_DESCRIPTION
- Added optional AWS_SESSION_TOKEN to BedrockService
- Support creating status events via the API

### Changed

- Moved tool call success log to DEBUG level
- Optimized canrep to not generate a draft in strict mode if no canrep candidates found
- Removed `acknowledged_event_offset` from status events
- Removed `last_known_event_offset` from `LoadedContext.interaction`

### Fixed

- Fixed presentation of missing API keys for built-in NLP services
- Improvements to canned response generation
- Fixed bug with null journey paths in some cases
- Fixed tiny bug with terminal nodes in journey node selection
- Fixed evaluations not showing properly after version upgrade

## [3.0.1] - 2025-08-16

### Changed

- Move tool call success log to DEBUG level

### Fixed

- Fix tool-based variable not enabling the associated tool on the server
- Fix authorization errors throwing 500 instead of 403
- Changed OpenAI LLM request operation level to TRACE to fix evaluation progress bars

## [3.0.0] - 2025-08-15

- Please see the announcement at https://parlant.io/blog/parlant-3-0-release

## [2.2.0] - 2025-05-20

### Added

- Add journeys
- Add of guideline properties evaluation
- Add automatic guideline action deduction when adding direct tool guidelines
- Added choices of invalid and missing tool parameters to tool insights

### Changed

- Make guideline action optional

## [2.1.2] - 2025-05-07

### Changed

- Remove interaction history from utterance recomposition prompt
- Use tool calls from the entire interaction for utterance field substitution
- Improve error handling and reporting with utterance rendering failures

### Fixed

- Always reason about utterance selection to improve performance

## [2.1.1] - 2025-04-30

### Fixed

- Fixed rendering relationships in CLI
- Fixed parlant client using old imports from python client SDK

## [2.1.0] - 2025-04-29

### Added

- ToolParameterOptions.choice_provider can now access ToolContext
- Added utterance/draft toggle in the integrated UI
- Added new guideline relationship: Dependency
- Added tool relationships and the OVERLAP relationship
- Added the 'overlap' property to tools. By default, tools will be assumed not to overlap with each other, simplifying their evaluation at runtime.
- Introduce ToolBatchers
- Introduce Journey

### Changed

- Improved tool calling efficiency by adjusting the prompt to the tool at hand
- Revised completion schema (ARQs) for tool calling
- Utterances now follow a 2-stage process: draft + select
- Changed guest customer name to Guest

### Fixed

- Fixed deprioritized guidelines always being skipped
- Fixed agent creation with tags
- Fixed client CLI exit status when encountering an error
- Fixed agent update

### Known Issues

- OpenAPI tool services sometimes run into issues due to a version update in aiopenapi3

## [2.0.0] - 2025-04-09

### Added

- Improved tool parameter flexibility: custom types, Pydantic models, and annotated ToolParameterOptions
- Allow returning a new (modified) container in modules using configure_module()
- Added Tool Insights with tool parameter options
- Added support for default values for tool parameters in tool calling
- Added WebSocket logger feature for streaming logs in real time
- Added a log viewer to the sandbox UI
- Added API and CLI for Utterances
- Added support for the --migrate CLI flag to enable seamless store version upgrades during server startup
- Added clear rate limit error logs for NLP adapters
- Added enabled/disabled flag for guidelines to facilitate experimentation without deletion
- Allow different schematic generators to adjust incoming prompts in a structured manner
- Added tags to context variables, guidelines, glossary and agents
- Added guideline matching strategies
- Added guideline relationships
- Added support for tool parameters choice provider using the tool context as argument

### Changed

- Made the message generator slightly more polite by default, following user feedback
- Allow only specifying guideline condition or action when updating guideline from CLI
- Renamed guideline proposer with guideline matcher

### Fixed

- Lowered likelihood of the agent hallucinating facts in fluid mode
- Lowered likelihood of the agent offering services that were not specifically mentioned by the business

## [1.6.2] - 2025-01-29

### Fixed

- Fix loading DeepSeek service during server boot

## [1.6.1] - 2025-01-20

### Fixed

- Fix ToolCaller not getting clear information on a parameter being optional
- Ensure ToolCaller only calls a tool if all required args were given
- Improve valid JSON generation likelihood in MessageEventGenerator
- Improve ToolCaller's ability to correctly run multiple tools at once

## [1.6.0] - 2025-01-19

### Added

- Add shot creation helper functions under Shot
- Add ContextEvaluation in MessageEventGenerator
- Add a log command under client CLI for streaming logs
- Add engine lifecycle hooks

### Changed

- Split vendor dependencies to extra packages to avoid reduce installation time
- Modified ToolCaller shot schema
- Disable coherence and connection checking by default in the CLI for now

### Fixed

- Improved GuidelineProposer's ability to handle compound actions
- Improved GuidelineProposer's ability to distinguish between a fulfilled and unfulfilled action
- Improved GuidelineProposer's ability to detect a previously applied guideline's application to new information
- Reduced likelihood of agent offering hallucinated services
- Fix ToolCaller false-negative argument validation from int to float
- Fix ToolCaller accuracy
- Fix ToolCaller making up argument values when it doesn't have them
- Fix some cases where the ToolCaller also calls a less-fitting tool
- Fix mistake in coherence checker few shots
- Fix markdown tables in sandbox UI
- Fix wrong import of RateLimitError
- Fix PluginServer validation for optional tool arguments when they're passed None
- Fix utterances sometimes not producing a message

## [1.5.1] - 2025-01-05

### Fixed

- Fix server CLI boot

## [1.5.1] - 2025-01-05

### Fixed

- Fix server CLI boot

## [1.5.0] - 2025-01-04

### Added

- Add DeepSeek provider support (via DeepSeekService)

### Changed

- Change default home dir from runtime-data to parlant-data

### Fixed

- Fix tool-calling test
- Fix HuggingFace model loading issues

## [1.4.3] - 2025-01-02

### Fixed

- Upgraded dependency "tiktoken" to 0.8.0 to fix installation errors on some environments

## [1.4.2] - 2024-12-31

### Fixed

- Fix race condition in JSONFileDocumentDatabase when deleting or updating documents

## [1.4.1] - 2024-12-31

### Changed

- Remove tool metadata from prompts - agents are now only aware of the data itself

### Fixed

- Fix tool calling in scenarios where a guideline has multiple tools where more than one should run

## [1.4.0] - 2024-12-31

### Added

- Support custom plugin data for PluginServer
- Allow specifying custom logger ID when creating loggers
- Add 'hosted' parameter to PluginServer, for running inside modules

### Fixed

- Fix the tool caller's few shots to include better rationales and arguments.

## [1.3.1] - 2024-12-27

### Changed

- Return event ID instead of trace ID from utterance API
- Improve and normalize entity update messages in client CLI

## [1.3.0] - 2024-12-26

### Added

- Add manual utterance requests
- Refactor few-shot examples and allow adding more examples from a module
- Allow tapping into the PluginServer FastAPI app to provide additional custom endpoints
- Support for union parameters ("T | None") in tool functions

### Changed

- Made all stores thread-safe with reader/writer locks
- Reverted GPT version for guideline connection proposer to 2024-08-06
- Changed definition of causal connection to take the source's when statement into account. The connection proposer now assumes the source's condition is true when examining if it entails other guideline.

### Fixed

- Fix 404 not being returned if a tool service isn't found
- Fix having direct calls to asyncio.gather() instead of safe_gather()

### Removed

- Removed connection kind (entails / suggests) from the guideline connection proposer and all places downstream. the connection_kind argument is no longer needed or supported for all guideline connections.

## [1.2.0] - 2024-12-19

### Added

- Expose deletion flag for events in Session API

### Changed

- Print traceback when reporting server boot errors
- Make cancelled operations issue a warning rather than an error

### Fixed

- Fixed tool calling with optional parameters
- Fixed sandbox UI issues with message regeneration and status icon
- Fixed case where guideline is applied due to condition being partially applied

### Removed

None

## [1.1.0] - 2024-12-18

### Added

- Customer selection in sandbox Chat UI
- Support tool calls with freshness rules for context variables
- Add support for loading external modules for changing engine behavior programmatically
- CachedSchematicGenerator to run the test suite more quickly
- TransientVectorDatabase to run the test suite more quickly

### Changed

- Changed model path for Chroma documents. You may need to delete your `runtime-data` dir.

### Fixed

- Improve handling of partially fulfilled guidelines

### Removed

None
