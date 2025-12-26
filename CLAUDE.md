This is the main repo of Parlant (https://parlant.io).

Parlant is a Python based agent framework. Its core strengths:

1. It allows you to create compliant and controlled AI agents for customer-facing use cases
2. It provides many conversational management features out of the box
3. It's built for enterprise, large-scale use cases, where SLAs, stability and security are paramount

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
