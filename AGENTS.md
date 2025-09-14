# Repository Guidelines

## Project Structure & Module Organization
- Source: `src/parlant/` organized by domain:
  - `core/` (engines, services, persistence, NLP, entities)
  - `api/` (FastAPI app and chat endpoints)
  - `adapters/` (DB, vector DB, loggers)
  - `bin/` (CLI entry points: server/client)
  - `sdk.py` (high-level SDK surface)
- Tests: `tests/` with `core/{stable,unstable}`, `api/`, `sdk/`, `e2e/`.
- Tooling: `pyproject.toml` (Poetry), `ruff.toml`, `mypy.ini`, `pytest.ini`, `scripts/`.

## Build, Test, and Development Commands
- Install (all extras): `poetry install --all-extras`
- Lint (Ruff + format check): `poetry run ruff check . && poetry run ruff format --check`
- Type check: `poetry run mypy`
- Auto-format: `poetry run ruff format .`
- Run tests (quick): `poetry run pytest -q`
- Core stable suite: `poetry run pytest tests/core/stable -q`
- Start server: `poetry run parlant-server` (defaults to `http://localhost:8800`)
- CLI client: `poetry run parlant --help`
- CI-parity scripts: `python scripts/lint.py --mypy --ruff`, `python scripts/install_packages.py`

## Coding Style & Naming Conventions
- Python 3.10+; line length 100; 4-space indent; double quotes (Ruff formatter).
- Type hints required; `mypy` runs in `strict` mode; keep `py.typed` coverage.
- Module/file names: `snake_case.py`; classes: `PascalCase`; functions/vars: `snake_case`.
- Prefer explicit imports and small, focused modules under `core/` subpackages.

## Testing Guidelines
- Framework: `pytest` (+ `pytest-asyncio`, `pytest-bdd`, `pytest-cov`).
- Place deterministic tests in `tests/core/stable`; exploratory in `core/unstable`; E2E in `tests/e2e`.
- Name files `test_*.py`; use `-q` for concise runs; add coverage: `poetry run pytest --cov=parlant`.
- Provide fixtures in `tests/conftest.py`; prefer async tests where applicable.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤72 chars). Examples: `Add __repr__ to Guideline`, `Fix healthcare example`.
- PRs: clear description, link issues (e.g., `Fixes #123`), include test updates, and note behavioral/API changes.
- Ensure `ruff`, `mypy`, and `pytest` pass locally before requesting review.

## Security & Configuration Tips
- Set required keys via env vars (e.g., `OPENAI_API_KEY`) for tests/server.
- Do not commit secrets; use `.env` locally with `python-dotenv` and GitHub Actions secrets in CI.
