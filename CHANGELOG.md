# Changelog

All notable changes to Parlant will be documented here.

## [Unreleased]
TBD


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
- Return event ID instead of correlation ID from utterance API
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
- Add support for loading external modules for changing engine behavior programatically
- CachedSchematicGenerator to run the test suite more quickly
- TransientVectorDatabase to run the test suite more quickly

### Changed

- Changed model path for Chroma documents. You may need to delete your `runtime-data` dir.

### Fixed

- Improve handling of partially fulfilled guidelines

### Removed

None
