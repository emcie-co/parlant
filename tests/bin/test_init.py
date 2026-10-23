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

import os
import stat
import subprocess
import sys
from pathlib import Path

from click.testing import CliRunner
import pytest

import parlant.bin.cli as cli_module
from parlant.bin.cli import cli
from parlant.bin.scaffold import PackageManager, detect_package_managers


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_that_the_package_manager_menu_reports_what_is_installed() -> None:
    detected = detect_package_managers()

    assert [d.manager for d in detected] == [
        PackageManager.UV,
        PackageManager.PIP,
        PackageManager.CONDA,
        PackageManager.POETRY,
    ]

    # Whatever is running these tests has a Python, so pip is always an option.
    pip = next(d for d in detected if d.manager is PackageManager.PIP)
    assert pip.installed is True

    # The rest are reported honestly, whichever way this machine is set up.
    for detection in detected:
        assert isinstance(detection.installed, bool)


def test_that_init_scaffolds_a_project_without_prompting_when_told_which_manager(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["init", "--package-manager", "uv", "--no-git"])

        assert result.exit_code == 0, result.output

        for expected in (
            "main.py",
            "config.py",
            "domains/__init__.py",
            "tests/__init__.py",
            "README.md",
            ".env.example",
            ".gitignore",
        ):
            assert Path(expected).is_file(), f"missing {expected}"

        # parlant-data is runtime state, never committed.
        assert "parlant-data" in Path(".gitignore").read_text()


def test_that_the_default_scaffold_carries_no_example_domain(
    runner: CliRunner,
) -> None:
    # The default is for someone who already knows what they're building: the
    # structure and the conventions, none of the example content.
    with runner.isolated_filesystem():
        assert runner.invoke(cli, ["init", "--package-manager", "uv", "--no-git"]).exit_code == 0

        assert not Path("domains/orders").exists()
        assert not Path("domains/disputes").exists()

        # Still runnable and still testable, or it isn't a project yet.
        assert "create_server" in Path("main.py").read_text()
        assert Path("tests/__init__.py").is_file()
        assert not list(Path("tests").glob("test_*.py"))


def test_that_the_retail_scaffold_carries_the_orders_and_disputes_domains(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", "retail"],
        )

        assert result.exit_code == 0, result.output

        for expected in (
            "domains/orders/domain.py",
            "domains/orders/tools.py",
            "domains/orders/service.py",
            "domains/disputes/domain.py",
            "domains/disputes/tools.py",
            "domains/disputes/service.py",
            "tests/domains/orders/test_orders.py",
            "tests/domains/disputes/test_disputes.py",
        ):
            assert Path(expected).is_file(), f"missing {expected}"


def test_that_every_templated_canned_response_declares_what_it_needs(
    runner: CliRunner,
) -> None:
    # A template whose fields nobody staged is still offered to the agent, with
    # its placeholders unfilled, unless it declares them as dependencies.
    with runner.isolated_filesystem():
        assert (
            runner.invoke(
                cli,
                ["init", "--package-manager", "uv", "--no-git", "--scaffold", "retail"],
            ).exit_code
            == 0
        )

        for path in ("domains/orders/domain.py", "domains/disputes/domain.py"):
            source = Path(path).read_text()

            for block in source.split("create_canned_response(")[1:]:
                template = block.split(")")[0]

                if "{{" not in template:
                    continue

                assert (
                    "field_dependencies="
                    in block[: block.index("await") if "await" in block else len(block)]
                ), f"{path}: a templated canned response declares no field_dependencies"


def test_that_the_retail_example_enforces_the_photo_in_the_tool_not_only_in_the_policy(
    runner: CliRunner,
) -> None:
    # The point of the example: a prompted step can be talked around, so the
    # requirement that protects the claim is also enforced where the dispute
    # is actually filed.
    with runner.isolated_filesystem():
        assert (
            runner.invoke(
                cli,
                ["init", "--package-manager", "uv", "--no-git", "--scaffold", "retail"],
            ).exit_code
            == 0
        )

        tools = Path("domains/disputes/tools.py").read_text()
        policies = Path("domains/disputes/domain.py").read_text()

        assert "requires_photo" in tools
        assert "first_image_url" in tools
        assert "attach a photo" in policies


def test_that_the_retail_scaffold_shows_every_kind_of_declaration(
    runner: CliRunner,
) -> None:
    # The retail scaffold is the worked example, so each thing a domain can
    # declare on an agent has to appear at least once — otherwise there is
    # nothing to copy from.
    with runner.isolated_filesystem():
        assert (
            runner.invoke(
                cli,
                ["init", "--package-manager", "uv", "--no-git", "--scaffold", "retail"],
            ).exit_code
            == 0
        )

        declared = "\n".join(
            Path(path).read_text()
            for path in ("main.py", "domains/orders/domain.py", "domains/disputes/domain.py")
        )

        for declaration in (
            "create_policy",
            "create_observation",
            "create_canned_response",
            "create_term",
            "create_variable",
        ):
            assert declaration in declared, f"the example never uses {declaration}"


@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_every_scaffold_starts_at_debug_log_level(
    runner: CliRunner,
    scaffold: str,
) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
        )

        assert result.exit_code == 0, result.output
        assert "log_level=p.LogLevel.DEBUG" in Path("config.py").read_text()


@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_every_scaffold_loads_its_env_file(
    runner: CliRunner,
    scaffold: str,
) -> None:
    # The README tells the reader to put their API key in .env, so something
    # has to read it. Without this the key is silently absent and the first
    # model call fails.
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
        )

        assert result.exit_code == 0, result.output

        config = Path("config.py").read_text()

        assert "load_dotenv" in config
        assert "python-dotenv" in Path("pyproject.toml").read_text()


@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_every_scaffold_routes_models_through_openrouter(
    runner: CliRunner,
    scaffold: str,
) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
        )

        assert result.exit_code == 0, result.output
        assert '"*": "openrouter"' in Path("config.py").read_text()
        assert "OPENROUTER_API_KEY" in Path(".env.example").read_text()


def test_that_init_writes_start_scripts_for_unix_and_windows(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli, ["init", "--package-manager", "uv", "--no-git"]).exit_code == 0

        unix = Path("run.sh")
        windows = Path("run.cmd")

        assert unix.is_file() and windows.is_file()
        # The unix script has to be runnable without a chmod first.
        assert unix.stat().st_mode & stat.S_IXUSR


@pytest.mark.parametrize(
    ("manager", "expected_command", "dependency_file"),
    [
        ("uv", "uv run", "pyproject.toml"),
        ("pip", "python", "requirements.txt"),
        ("conda", "conda run", "environment.yml"),
        ("poetry", "poetry run", "pyproject.toml"),
    ],
)
def test_that_the_start_script_and_dependencies_match_the_chosen_manager(
    runner: CliRunner,
    manager: str,
    expected_command: str,
    dependency_file: str,
) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["init", "--package-manager", manager, "--no-git"])

        assert result.exit_code == 0, result.output
        assert Path(dependency_file).is_file()
        assert expected_command in Path("run.sh").read_text()
        assert expected_command in Path("run.cmd").read_text()


@pytest.mark.parametrize("manager", ["uv", "pip", "conda", "poetry"])
def test_that_every_package_manager_gets_pytest_configured_and_installed(
    runner: CliRunner,
    manager: str,
) -> None:
    # The generated tests are async and carry no per-test marker, so every
    # project needs asyncio_mode set somewhere pytest reads — and needs
    # pytest-asyncio actually installed by its own install command.
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", manager, "--no-git", "--scaffold", "retail"],
        )

        assert result.exit_code == 0, result.output

        configured = "\n".join(
            Path(name).read_text()
            for name in ("pyproject.toml", "pytest.ini")
            if Path(name).exists()
        )
        assert "asyncio_mode" in configured

        declared = "\n".join(
            Path(name).read_text()
            for name in ("pyproject.toml", "requirements.txt", "environment.yml")
            if Path(name).exists()
        )
        assert "pytest-asyncio" in declared


def test_that_init_confirms_before_scaffolding_into_the_current_directory(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "_is_interactive", lambda: True)

    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git"],
            input="y\n",
        )

        assert result.exit_code == 0, result.output
        assert Path("main.py").is_file()


def test_that_declining_the_confirmation_writes_nothing(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli_module, "_is_interactive", lambda: True)

    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git"],
            input="n\n",
        )

        assert result.exit_code != 0
        assert not Path("main.py").exists()


def test_that_init_does_not_confirm_when_given_a_destination(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An explicit destination is already the user saying where; asking again
    # would just be noise, and would hang a non-interactive caller.
    monkeypatch.setattr(cli_module, "_is_interactive", lambda: True)

    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["init", "my-agent", "--package-manager", "uv", "--no-git"])

        assert result.exit_code == 0, result.output
        assert Path("my-agent/main.py").is_file()


def test_that_init_refuses_a_non_empty_directory_without_force(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        Path("main.py").write_text("# my own work\n")

        result = runner.invoke(cli, ["init", "--package-manager", "uv", "--no-git"])

        assert result.exit_code != 0
        assert "--force" in result.output
        assert Path("main.py").read_text() == "# my own work\n"


def test_that_init_does_not_nest_a_git_repository(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        subprocess.run(["git", "init", "--quiet"], check=True)

        result = runner.invoke(cli, ["init", "--package-manager", "uv"])

        assert result.exit_code == 0, result.output
        # It says what it skipped rather than nesting a second repository.
        assert "git" in result.output.lower()
        assert not Path("main.py/.git").exists()


def test_that_init_accepts_a_target_directory(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["init", "my-agent", "--package-manager", "uv", "--no-git"])

        assert result.exit_code == 0, result.output
        assert Path("my-agent/main.py").is_file()


@pytest.mark.slow
@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_the_scaffolded_project_is_lint_clean(
    runner: CliRunner,
    tmp_path: Path,
    scaffold: str,
) -> None:
    # A brand new project shouldn't greet its owner with lint errors the
    # moment they wire up their own tooling.
    project = tmp_path / "generated"

    result = runner.invoke(
        cli,
        ["init", str(project), "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
    )
    assert result.exit_code == 0, result.output

    completed = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "."],
        cwd=project,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.slow
@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_the_scaffolded_project_passes_its_own_tests(
    runner: CliRunner,
    tmp_path: Path,
    scaffold: str,
) -> None:
    # The scaffold is what every new project starts from, so it has to be
    # valid Python whose tests pass — not just a set of files that exist.
    # The basic scaffold has no project code yet, so pytest correctly reports
    # that it collected nothing (exit code 5); inventing a test to turn that
    # into a green run would only teach the reader to write hollow tests.
    project = tmp_path / "generated"
    project.mkdir(exist_ok=True)

    result = runner.invoke(
        cli,
        ["init", str(project), "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
    )
    assert result.exit_code == 0, result.output

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=project,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(project)},
    )

    expected = (0, 5) if scaffold == "basic" else (0,)

    assert completed.returncode in expected, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    ("manager", "install_marker"),
    [
        ("pip", "pip install"),
        ("conda", "conda env create"),
        ("poetry", "poetry install"),
    ],
)
def test_that_the_start_script_installs_dependencies_before_running(
    runner: CliRunner,
    manager: str,
    install_marker: str,
) -> None:
    # ./run.sh (and run.cmd) must work on a fresh checkout without a separate
    # install step: they ensure dependencies before launching main.py. (uv is
    # turnkey already — `uv run` syncs on its own.)
    with runner.isolated_filesystem():
        assert runner.invoke(cli, ["init", "--package-manager", manager, "--no-git"]).exit_code == 0

        for script in ("run.sh", "run.cmd"):
            content = Path(script).read_text()
            assert install_marker in content, f"{script}: no dependency install step"
            assert content.index(install_marker) < content.index("main.py"), (
                f"{script}: the install step must precede running main.py"
            )


@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_every_scaffold_writes_an_agents_file_naming_parlant(
    runner: CliRunner,
    scaffold: str,
) -> None:
    # A coding agent opening this repo has no way to know the framework from
    # the file layout alone: domains/, tools/ and policies read like any other
    # Python project. AGENTS.md is the one file it is conventionally going to
    # look at, so that is where the repo says what it is.
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
        )

        assert result.exit_code == 0, result.output

        agents = Path("AGENTS.md").read_text()

        assert "Parlant" in agents
        assert "https://parlant.io" in agents


@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_the_agents_file_tells_coding_agents_to_consult_the_docs_command(
    runner: CliRunner,
    scaffold: str,
) -> None:
    # The point of the file: an agent that guesses at Parlant's idioms writes
    # plausible-looking code that fights the engine. `parlant docs query` is
    # how it checks instead of guessing, and context is what makes the answer
    # specific to what the user is actually building.
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
        )

        assert result.exit_code == 0, result.output

        agents = Path("AGENTS.md").read_text()

        assert "parlant docs query" in agents
        assert "--context" in agents
        assert agents.count("parlant docs query") >= 5, (
            "the file should show several worked examples, not just name the command"
        )


@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_the_agents_file_maps_each_kind_of_change_to_its_construct(
    runner: CliRunner,
    scaffold: str,
) -> None:
    # A coding agent reaches for whatever construct it used last. Asked to
    # change how the agent sounds, it writes another policy, when tone and
    # wording belong to canned responses. An index of which construct does
    # which job lets it pick right before it writes anything.
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
        )

        assert result.exit_code == 0, result.output

        agents = Path("AGENTS.md").read_text()

        assert "Which construct for which job" in agents
        section = agents[agents.index("Which construct for which job") :]
        section = section[: section.index("\n## ")] if "\n## " in section else section

        for job, construct in [
            ("tone of voice and wording choices", "canned response"),
            ("identity", "instructions"),
            ("in a situation", "policy"),
            ("recognize a situation", "observation"),
            ("fetch data", "tool"),
            ("how a word is read", "glossary term"),
            ("per-customer facts", "variable"),
        ]:
            line = next((line for line in section.splitlines() if job in line.lower()), None)
            assert line is not None, f"no row for {job!r}"
            assert construct in line.lower(), f"{job!r} should map to {construct!r}: {line}"

        assert "greeting" not in section.lower()
        assert "https://parlant.io/docs/core-concepts/which-construct" in section
        assert "however small" in agents


@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_the_agents_file_requires_a_docs_query_before_any_parlant_code_change(
    runner: CliRunner,
    scaffold: str,
) -> None:
    # Coding agents skip the docs when they judge a change too small to be a
    # "design decision": a wording fix written as a policy, a price table
    # copied from another project. The rule has to be mechanical (any SDK call
    # or server setting) and come first, before the agent starts writing.
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
        )

        assert result.exit_code == 0, result.output

        agents = Path("AGENTS.md").read_text()

        rule = "before writing or changing any Parlant SDK call or server configuration"
        assert rule in agents
        assert "even when copying" in agents
        assert agents.index(rule) < agents.index("## Why guessing the code can go wrong here")


@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_the_agents_file_asks_the_agent_to_recommend_saving_a_memory(
    runner: CliRunner,
    scaffold: str,
) -> None:
    # AGENTS.md is read once per session and is easy to stop applying midway.
    # A memory the user approves follows the coding agent into later sessions
    # and other Parlant projects.
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
        )

        assert result.exit_code == 0, result.output

        agents = Path("AGENTS.md").read_text()

        start = agents.index("IMPORTANT: Recommend")
        recommendation = agents[start : agents.index("\n\n", start)]
        assert "memory" in recommendation
        assert "parlant docs query" in recommendation


@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_the_agents_file_explains_hosting_several_agents_on_one_server(
    runner: CliRunner,
    scaffold: str,
) -> None:
    # A site often needs more than one agent (say, one per page). Every
    # scaffold creates exactly one, so without a note a coding agent tends to
    # copy the whole project per agent instead of adding a create_agent call,
    # and does not know the id it picks is what the chat widget selects by.
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
        )

        assert result.exit_code == 0, result.output

        agents = Path("AGENTS.md").read_text()

        assert "several agents" in agents.lower()
        assert "create_agent" in agents
        assert "install(agent)" in agents
        assert "agentId" in agents


@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_the_agents_file_says_a_failed_reload_stops_the_server(
    runner: CliRunner,
    scaffold: str,
) -> None:
    # Hot reload re-runs main.py on every save, so a coding agent assumes the
    # server is always up. An edit that fails at startup ends the process
    # instead, and the agent goes on testing against nothing.
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
        )

        assert result.exit_code == 0, result.output

        agents = Path("AGENTS.md").read_text()

        assert "hot reload" in agents.lower()
        assert "start it again" in agents


@pytest.mark.parametrize("scaffold", ["basic", "retail"])
def test_that_the_agents_file_asks_the_user_to_sign_in_to_the_dashboard_first(
    runner: CliRunner,
    scaffold: str,
) -> None:
    # A coding agent cannot see the dashboard. Once the user has signed the CLI
    # in, it can check for itself whether the dashboard sees the server, and
    # read the project's agents and sessions, instead of asking the user to
    # look every time.
    with runner.isolated_filesystem():
        result = runner.invoke(
            cli,
            ["init", "--package-manager", "uv", "--no-git", "--scaffold", scaffold],
        )

        assert result.exit_code == 0, result.output

        agents = Path("AGENTS.md").read_text()

        assert "parlant dashboard auth login" in agents
        assert "parlant dashboard project check" in agents
        assert agents.index("parlant dashboard auth login") < agents.index(
            "parlant dashboard project check"
        )


def test_that_init_does_not_overwrite_an_existing_agents_file_without_force(
    runner: CliRunner,
) -> None:
    # Most repos that gain a Parlant project already have an AGENTS.md, and it
    # is hand-written. Scaffolding over it would be the single most annoying
    # thing init could do, so it joins the ordinary clash list.
    with runner.isolated_filesystem():
        Path("AGENTS.md").write_text("# My own instructions\n")

        result = runner.invoke(cli, ["init", "--package-manager", "uv", "--no-git"])

        assert result.exit_code != 0
        assert "AGENTS.md" in result.output
        assert Path("AGENTS.md").read_text() == "# My own instructions\n"
