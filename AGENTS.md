This is the main repo of Parlant (https://parlant.io).

Parlant is a vibe-coding platform for chat agents that talk to customers. Someone describes how their agent should behave, and their coding agent implements it against Parlant's Python SDK. Parlant provides the rest: an engine that keeps the agent on those rules in every conversation, plus the parts every customer-facing agent needs — a chat widget, human handoff, testing and hosting.

Three things follow from that, and they apply to every change you make here:

1. Agents built on Parlant must stay compliant and controlled. A rule the engine can quietly drop is a bug, not a tuning problem.
2. Conversational management is meant to come out of the box. If a capability only works when the integrator writes glue code around it, it isn't finished.
3. It has to support enterprise, large-scale use cases, where SLAs, stability and security are paramount. That is a bar every change has to clear, not the audience Parlant is built around.

Most people reach Parlant through their coding agent rather than by typing calls themselves. That agent cannot ask what a flag does or what an error was getting at — it reads the docstring, the `--help` text and the error string, and acts on them. Write those as part of the API: say what to do next, not only what went wrong.

The repo's structure follows the Hexagonal Architecture (Ports and Adapters) approach.

- src/parlant
  - core: Core framework code
  - adapters: Implementations of interfaces using 3rd party tools
  - api: REST API layer using FastAPI. Uses modules from core/
- tests: all tests for the project. Structure strives to mirror that which is under src/parlant.

General Coding Instructions:

- Always ensure you stick to Hexagonal Architecture patterns in line with how they're used in this codebase.
- Every time you add something, look for similar things in the codebase and ensure you follow the coding style.
- We use MyPy on strict mode. Every parameter needs to be type-annotated. Every function's result too.
- If you need to add a test for something, first say where you plan to add it and ask for confirmation.
- We follow TDD. When you make a change, first create a failing test. Once it fails, implement just enough so it passes.
- We avoid the mock-ist approach to testing. We follow classical TDD as much as possible, unless to avoid very slow tests in particular component tests... SDK tests never use mocks. Other tests may mock specific LLM or embedding requests, especially to generate edge-cases deterministically. Otherwise, we always follow classical testing by integrating real components. This makes for a more stable and robust test suite.
- If you need to test classes/methods in sdk.py (or generally to test things that relate to engine behavior) make sure you inherit from SDKTest and understand how it works and how to use it.
- Test names should go "test*that*..." using clear names that explain the context, what is executed, and what is the expected result.
- You can run tests using pytest. Make sure you run "uv run pytest tests/path/to/test/file.py" while also specifying the test name that you need to run.

Always follow this plan when asked to code a feature or fix a bug:

1. Consider the codebase's structure
2. Describe your implementation plan, including:
   a. What tests you will write (test names + files they would live in)
   b. Why do you think the tests would initially fail
   c. Where you would plan to implement the code that would make the tests pass
3. Ask for plan confirmation. If you get feedback, revise your plan and ask for confirmation again until you get it.
4. Implement the tests first. Ask for confirmation and code review.
5. Once tests are approved, once again suggest your implementation plan for making them pass, and get plan review until confirmation.
6. Once your implementation plan is confirmed, go ahead with implementing the code to pass them.
7. Make sure to format all of the files you changed using ruff (it is installed in the environment).
8. Run `uv run python scripts/lint.py --mypy --ruff` to ensure your code has no lint issues.
9. Update `CHANGELOG.md` under `## [Unreleased]` as part of the same commit, in the matching section (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`). Write it for the person upgrading: what changed and what it means for them, not which files you touched. A change nobody outside this repo can observe — a refactor, a test, a comment — needs no entry.
