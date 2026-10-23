"""Create a project; a managed one is then set up the way the dashboard sets it up.

A managed project is created from a repository and a branch. Its setup check
starts on its own; the command follows it, shows what it found, and saves any
commands, environment and secrets through the project's setup."""

from __future__ import annotations

import os
import re
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import click

from parlant.bin.dashboard.auth import authenticate, fetch_session
from parlant.bin.dashboard.client import project_path, request, require_ok
from parlant.bin.dashboard.commands._shared import project_kind, runtime_for
from parlant.bin.dashboard.commands.projects import project
from parlant.bin.dashboard.errors import DashboardError
from parlant.bin.dashboard.output import emit_json, json_mode, json_option, say
from parlant.bin.dashboard.runtime import DashboardRuntime
from parlant.bin.help import examples

CHECK_DEADLINE_SECONDS = 180

CHECK_POLL_SECONDS = 2.0

VERDICTS = {
    "standard": "looks like a standard Parlant project",
    "review": "this project has its own setup; review the marked fields",
    "unknown": "could not work out how to run this project",
    "none": "could not find a Parlant server to start",
}

SETUP_PAGE = "Overview, with this project selected"

FIELD_LABELS = {"buildCommand": "build command", "runCommand": "run command"}


@project.command(
    "create",
    epilog=examples(
        "parlant dashboard project create",
        "parlant dashboard project create --kind local --name my-agent --yes",
        "parlant dashboard project create --kind managed --repo example/agent --branch main --run-command 'python main.py' --yes",
    ),
)
@click.option("--name", help="Project name.")
@click.option(
    "--kind",
    type=click.Choice(["managed", "local", "self-hosted"]),
    help="Who runs the server: managed (Parlant builds and runs it from your "
    "repository), local (you run it on your machine), or self-hosted.",
)
@click.option("--repo", help="GitHub repository, e.g. example/agent.")
@click.option("--installation", help="Workspace GitHub installation ID.")
@click.option("--branch", help="Repository branch.")
@click.option("--run-command", help="Run command, including the chosen entrypoint.")
@click.option("--build-command", help="Build command; omitted uses the detected default.")
@click.option(
    "--build-env",
    multiple=True,
    metavar="KEY=VALUE",
    help="An environment variable for the build step. Repeatable.",
)
@click.option(
    "--run-env",
    multiple=True,
    metavar="KEY=VALUE",
    help="An environment variable for the running server. Repeatable. Use "
    "--secret for anything that must not be stored as plain text.",
)
@click.option(
    "--secret",
    "secret_refs",
    multiple=True,
    metavar="KEY=ENVIRONMENT_VARIABLE",
    help="Read a runtime secret from a local environment variable.",
)
@click.option("--request-id", help="Reuse the printed 24-hex ID to resume an interrupted creation.")
@click.option("--yes", is_flag=True, help="Confirm supplied settings without interactive prompts.")
@json_option
@click.pass_context
def create_project(
    ctx: click.Context,
    name: str | None,
    kind: str | None,
    repo: str | None,
    installation: str | None,
    branch: str | None,
    run_command: str | None,
    build_command: str | None,
    build_env: tuple[str, ...],
    run_env: tuple[str, ...],
    secret_refs: tuple[str, ...],
    request_id: str | None,
    yes: bool,
) -> None:
    """Collect setup, review it, and create the configured project."""
    if json_mode():
        if not yes:
            raise DashboardError("--json needs --yes: it never prompts.")
        if kind == "managed":
            raise DashboardError(
                "--json creates local and self-hosted projects. Create managed projects "
                "without it: their setup check is interactive output."
            )
    created = create(
        runtime_for(ctx),
        name=name,
        kind=kind,
        repo=repo,
        installation=installation,
        branch=branch,
        run_command=run_command,
        build_command=build_command,
        build_env=build_env,
        run_env=run_env,
        secret_refs=secret_refs,
        request_id=request_id,
        yes=yes,
    )
    if json_mode() and created is not None:
        setup = created.get("setup")
        emit_json(
            {
                "id": created.get("id"),
                "name": created.get("name"),
                "projectKind": created.get("projectKind"),
                "setupStatus": setup.get("status") if isinstance(setup, dict) else None,
            }
        )


def create(
    runtime: DashboardRuntime,
    *,
    name: str | None,
    kind: str | None,
    repo: str | None = None,
    installation: str | None = None,
    branch: str | None = None,
    run_command: str | None = None,
    build_command: str | None = None,
    build_env: tuple[str, ...] = (),
    run_env: tuple[str, ...] = (),
    secret_refs: tuple[str, ...] = (),
    request_id: str | None = None,
    yes: bool = False,
) -> dict[str, Any] | None:
    """Create the project, returning it, or None when the user cancels."""
    token = authenticate(runtime)
    session = fetch_session(runtime, token)
    granted = session.scope.split()
    _require_grant(granted, "projects.create")
    setup_options = any((run_command, build_command, build_env, run_env, secret_refs))
    if kind == "managed":
        _require_grant(granted, "projects.update")
    if request_id and not re.fullmatch(r"[0-9a-fA-F]{24}", request_id):
        raise DashboardError("--request-id must contain 24 hexadecimal characters.")
    identifier = request_id.lower() if request_id else secrets.token_hex(12)
    if request_id:
        recovered = request(runtime, "GET", f"/projects/{identifier}", access_token=token)
        if recovered.is_success:
            project = require_ok(recovered)
            say(f"Project created: {project['id']}")
            created: dict[str, Any] = project
            if setup_options:
                if project_kind(project) != "managed":
                    raise DashboardError(
                        "Repository, commands, environment and secrets options require --kind managed."
                    )
                _require_grant(granted, "projects.update")
                _set_up_created(
                    runtime,
                    project["id"],
                    _choices(build_command, run_command, build_env, run_env, secret_refs),
                    yes,
                )
            return created
        if recovered.status_code != 404:
            require_ok(recovered)
    kind = _value(kind, "Project kind", yes, choices=["managed", "local", "self-hosted"])
    if kind == "managed":
        _require_grant(granted, "projects.update")
    name = _value(name, "Project name", yes)
    readiness = require_ok(
        request(
            runtime,
            "GET",
            "/projects/creation-readiness",
            query={"projectKind": kind},
            access_token=token,
        )
    )
    if not readiness.get("ready"):
        raise DashboardError(
            f"Project creation is blocked: {readiness.get('blocker') or 'workspace not ready'}. Complete workspace setup in the dashboard and retry."
        )
    payload: dict[str, Any] = {"id": identifier, "name": name, "projectKind": kind}
    choices: _Choices | None = None
    if kind == "managed":
        installation, repo = _repository(runtime, token, installation, repo, yes)
        token = authenticate(runtime)
        branches = require_ok(
            request(
                runtime,
                "GET",
                f"/github/installations/{installation}/repos/{quote(repo, safe='')}/branches",
                access_token=token,
            )
        )
        branch_names = [row["name"] for row in branches]
        branch = _value(branch, "Branch", yes, choices=branch_names)
        choices = _choices(build_command, run_command, build_env, run_env, secret_refs)
        payload["managedDeployment"] = {
            "githubInstallationId": installation,
            "githubRepoFullName": repo,
            "branch": branch,
        }
    elif repo or installation or branch or setup_options:
        raise DashboardError(
            "Repository, commands, environment and secrets options require --kind managed."
        )
    say(f"\nProject: {name}\nKind: {kind}")
    if kind == "managed":
        say(f"Repository: {repo}\nBranch: {branch}")
        say("After creation, a setup check reads the repository to find how to run it.")
    if not yes and not click.confirm("Create this project?", default=False):
        say("Canceled. No project was created.")
        return None
    say(f"Creation request ID: {identifier}")
    say(
        f"To resume after interruption, use --request-id {identifier} with the same name, "
        "repository and branch; setup options given with it are saved on the project."
    )
    token = authenticate(runtime)
    deadline = time.monotonic() + 120
    while True:
        response = request(runtime, "POST", "/projects", json_body=payload, access_token=token)
        body = require_ok(response)
        if response.status_code != 202:
            say(f"Project created: {body['id']}")
            if choices is not None:
                _set_up_created(runtime, body["id"], choices, yes)
            return dict(body)
        if time.monotonic() >= deadline:
            raise DashboardError(
                "Creation is still in progress. Resume using the request ID above; do not start another creation."
            )
        runtime.sleep(min(max(float(body.get("retryAfterSeconds", 2)), 1), 10))


def _repository(
    runtime: DashboardRuntime,
    token: str,
    installation: str | None,
    repo: str | None,
    noninteractive: bool,
) -> tuple[str, str]:
    connected_once = False
    while True:
        installed = require_ok(request(runtime, "GET", "/github/installations", access_token=token))
        selections = [
            (item["id"], row["fullName"])
            for item in installed
            for row in item.get("repositories", [])
            if not installation or item["id"] == installation
        ]
        matches = [row for row in selections if row[1] == repo] if repo else []
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise DashboardError(
                f"More than one connected installation exposes '{repo}'. "
                "Pass --installation <id> to choose which one to use."
            )
        if noninteractive:
            raise DashboardError(_unavailable_hint(repo))
        if repo or not selections:
            if connected_once and repo:
                _report_still_missing(repo, selections)
            _explain_connect()
            if not click.confirm("Open GitHub to connect a repository now?", default=True):
                raise DashboardError("No repository selected. No project was created.")
        else:
            index = _choose(
                "Repository",
                [f"{repository} ({install_id})" for install_id, repository in selections]
                + ["Connect GitHub"],
            )
            if index < len(selections):
                return selections[index]
        link = require_ok(request(runtime, "GET", "/github/install-url", access_token=token))
        click.echo("")
        click.echo("In the browser that opens:")
        click.echo("  1. Authorize the Parlant GitHub App for your account.")
        click.echo("  2. Choose the account or organization that owns your repository.")
        click.echo("  3. Grant it access to that repository — for an organization repository,")
        click.echo("     an organization owner may have to approve it.")
        click.echo(f"\nIf it does not open on its own, visit:\n  {link['url']}")
        runtime.open_browser(link["url"])
        click.prompt(
            "\nPress Enter once you have finished in the browser", default="", show_default=False
        )
        token = authenticate(runtime)
        connected_once = True


def _unavailable_hint(repo: str | None) -> str:
    """Explain, for a non-interactive run, why no repository is usable yet."""
    if repo:
        return (
            f"Repository '{repo}' is not available to this workspace. Install the Parlant "
            "GitHub App on the account that owns it and grant access to the repository — for "
            "an organization repository, an owner may need to approve it — then retry. Or drop "
            "--yes to connect GitHub interactively."
        )
    return (
        "No repository is connected to this workspace. Drop --yes to connect GitHub "
        "interactively, or install the Parlant GitHub App and grant it a repository first."
    )


def _explain_connect() -> None:
    click.echo(
        "\nTo deploy a managed project, the Parlant GitHub App must be installed on the "
        "account or organization that owns your repository, with access to that repository."
    )


def _report_still_missing(repo: str, selections: list[tuple[str, str]]) -> None:
    """After a connect attempt, say why the requested repo is still not visible."""
    click.echo(
        f"\n'{repo}' still is not available to this workspace. If it belongs to an "
        "organization, an owner has to grant the Parlant GitHub App access to that specific "
        "repository in the organization's GitHub App settings."
    )
    if selections:
        click.echo("Repositories the App can currently see:")
        for _, name in selections:
            click.echo(f"  - {name}")


def _value(
    value: str | None, label: str, noninteractive: bool, *, choices: list[str] | None = None
) -> str:
    if value:
        if choices is not None and value not in choices:
            raise DashboardError(f"{label} is not available in this workspace selection.")
        return value.strip()
    if noninteractive:
        raise DashboardError(f"{label} must be supplied with --yes.")
    if choices is not None:
        return choices[_choose(label, choices)]
    return str(click.prompt(label, type=click.STRING)).strip()


def _choose(label: str, values: list[str]) -> int:
    if not values:
        raise DashboardError(f"No {label.lower()} options are available.")
    for number, value in enumerate(values, 1):
        click.echo(f"{number}. {value}")
    return int(click.prompt(label, type=click.IntRange(1, len(values)))) - 1


def _pairs(entries: tuple[str, ...]) -> dict[str, str]:
    values: dict[str, str] = {}
    for entry in entries:
        key, separator, value = entry.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise DashboardError(
                "Environment entries must use KEY=VALUE with a valid environment variable name."
            )
        if key in values:
            raise DashboardError(f"Duplicate environment key: {key}.")
        values[key] = value
    return values


def _secret_value(environment_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", environment_name):
        raise DashboardError(
            "Secret sources must name local environment variables, not literal values."
        )
    if environment_name not in os.environ:
        raise DashboardError("A secret source environment variable is not set.")
    return os.environ[environment_name]


def _collect_pairs(label: str, values: dict[str, str], *, hidden: bool = False) -> None:
    while True:
        key = str(click.prompt(f"{label} key (empty to continue)", default="", show_default=False))
        if not key:
            return
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise DashboardError("Use a valid environment variable name.")
        values[key] = str(
            click.prompt(f"{label} value", hide_input=hidden, confirmation_prompt=hidden)
        )


@dataclass
class _Choices:
    """Settings given on the command line, saved after the setup check."""

    build_command: str | None
    run_command: str | None
    build_env: dict[str, str]
    run_env: dict[str, str]
    secrets: dict[str, str]


def _require_grant(granted: list[str], scope: str) -> None:
    if scope not in granted:
        raise DashboardError(
            f"This sign-in does not grant {scope}. Run `parlant dashboard auth login --reauthorize` "
            "and approve project creation."
        )


def _choices(
    build_command: str | None,
    run_command: str | None,
    build_env: tuple[str, ...],
    run_env: tuple[str, ...],
    secret_refs: tuple[str, ...],
) -> _Choices:
    build_values, run_values = _pairs(build_env), _pairs(run_env)
    secret_values = {key: _secret_value(env_name) for key, env_name in _pairs(secret_refs).items()}
    if set(run_values) & set(secret_values):
        raise DashboardError(
            "A key cannot appear in both run environment variables and runtime secrets."
        )
    return _Choices(build_command, run_command, build_values, run_values, secret_values)


def _set_up_created(
    runtime: DashboardRuntime, project_id: str, choices: _Choices, yes: bool
) -> None:
    """Set up a project that exists; a failure says so, keeping its exit status."""
    try:
        _set_up(runtime, project_id, choices, yes)
    except DashboardError as error:
        error.message = (
            f"{error.message} Project {project_id} was created; finish its setup in the "
            f"dashboard ({SETUP_PAGE}), then deploy with "
            f"`parlant dashboard --project {project_id} deploy`."
        )
        raise


def _set_up(runtime: DashboardRuntime, project_id: str, choices: _Choices, yes: bool) -> None:
    """Follow the automatic setup check, show its result, then save the user's settings."""
    click.echo("Checking the repository...")
    setup = _checked_setup(runtime, authenticate(runtime), project_id)
    _print_check(setup)
    settings = setup["settings"]
    run_command, build_command = choices.run_command, choices.build_command
    if not yes:
        suggested = [row["command"] for row in setup.get("runCommandSuggestions", [])]
        if not run_command and not settings["runCommand"] and len(suggested) > 1:
            run_command = suggested[_choose("Run command", suggested)]
        build_command = click.prompt(
            "Build command (empty uses platform default)",
            default=build_command if build_command is not None else settings["buildCommand"],
            show_default=False,
        )
        run_command = click.prompt(
            "Run command",
            default=run_command or settings["runCommand"],
            show_default=bool(run_command or settings["runCommand"]),
        )
        _collect_pairs("Build environment variable", choices.build_env)
        _collect_pairs("Run environment variable", choices.run_env)
        _collect_pairs("Runtime secret", choices.secrets, hidden=True)
        if set(choices.run_env) & set(choices.secrets):
            raise DashboardError(
                "A key cannot appear in both run environment variables and runtime secrets."
            )
    patch: dict[str, Any] = {}
    if build_command is not None and build_command.strip() != settings["buildCommand"]:
        patch["buildCommand"] = build_command.strip()
    if run_command and run_command.strip() != settings["runCommand"]:
        patch["runCommand"] = run_command.strip()
    for field, values in (("buildEnv", choices.build_env), ("runEnv", choices.run_env)):
        if values:
            kept = [row for row in settings[field] if row["name"] not in values]
            patch[field] = kept + [{"name": key, "value": value} for key, value in values.items()]
    run_env = patch.get("runEnv", settings["runEnv"])
    moved = [row["name"] for row in run_env if row["name"] in choices.secrets]
    if moved:
        # A name is either a run variable or a secret; the secret given here wins.
        patch["runEnv"] = [row for row in run_env if row["name"] not in choices.secrets]
        click.echo(f"Moving to runtime secrets: {', '.join(moved)}")
    token = authenticate(runtime)
    if patch:
        setup = require_ok(
            request(
                runtime,
                "PATCH",
                project_path(project_id, "managed-setup", "settings"),
                json_body=patch,
                access_token=token,
            )
        )
    for key, value in choices.secrets.items():
        setup = require_ok(
            request(
                runtime,
                "PUT",
                project_path(project_id, "managed-setup", "secrets", key),
                json_body={"value": value},
                access_token=token,
            )
        )
    if patch or choices.secrets:
        click.echo("Saved settings:")
        _print_settings(setup)
    _print_next_steps(project_id, setup)


def _checked_setup(runtime: DashboardRuntime, token: str, project_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + CHECK_DEADLINE_SECONDS
    while True:
        setup: dict[str, Any] = require_ok(
            request(
                runtime,
                "GET",
                project_path(project_id, "managed-setup"),
                access_token=token,
            )
        )
        if setup["check"]["status"] != "running" or time.monotonic() >= deadline:
            return setup
        runtime.sleep(CHECK_POLL_SECONDS)


def _print_check(setup: dict[str, Any]) -> None:
    check = setup["check"]
    status = check["status"]
    if status == "running":
        click.echo("The setup check is still running; its findings will appear on the setup page.")
    elif status == "failed":
        click.echo("The setup check failed before it could read the repository.")
    elif status == "stopped":
        click.echo("The setup check was stopped.")
    else:
        verdict = check.get("verdict", "")
        click.echo(f"Classification: {verdict} ({VERDICTS.get(verdict, 'checked')})")
        if check.get("summary"):
            click.echo(check["summary"])
    _print_settings(setup)
    for field in check.get("reviewFields", []):
        click.echo(
            f"Warning: Review the {FIELD_LABELS[field]}; this repository sets up its own way."
        )
    for suggestion in setup.get("suggestions", []):
        if suggestion["kind"] == "replace-field":
            what = f"{FIELD_LABELS[suggestion['field']]} '{suggestion['value']}'"
        else:
            what = f"{suggestion['scope']} environment variable {suggestion['name']}"
        click.echo(f"Warning: Suggested {what}{_reason(suggestion)}")
    saved = {row["name"] for row in setup.get("savedSecrets", [])}
    for secret in setup.get("expectedSecrets", []):
        if secret["name"] not in saved:
            click.echo(f"Warning: Expected secret {secret['name']} is not saved{_reason(secret)}")
    options = setup.get("runCommandSuggestions", [])
    if len(options) > 1:
        click.echo("Run command options:")
        for option in options:
            click.echo(f"  - {option['command']}{_reason(option)}")


def _print_settings(setup: dict[str, Any]) -> None:
    settings = setup["settings"]
    click.echo(f"Run command: {settings['runCommand'] or '(not set)'}")
    click.echo(f"Build command: {settings['buildCommand'] or '(platform default)'}")
    for label, field in (("Build environment", "buildEnv"), ("Run environment", "runEnv")):
        keys = ", ".join(row["name"] for row in settings[field])
        click.echo(f"{label} keys: {keys or '(none)'}")
    secrets = ", ".join(row["name"] for row in setup.get("savedSecrets", []))
    click.echo(f"Runtime secrets: {secrets or '(none)'}")


def _print_next_steps(project_id: str, setup: dict[str, Any]) -> None:
    page = f"the project's setup page in the dashboard ({SETUP_PAGE})"
    if setup["check"]["status"] != "done" or not setup["settings"]["runCommand"]:
        click.echo(f"Next: finish the setup on {page}, then deploy.")
    else:
        click.echo(f"Next: review the settings on {page}.")
    click.echo(f"To deploy from here: parlant dashboard --project {project_id} deploy")


def _reason(entry: dict[str, Any]) -> str:
    return f" ({entry['reason']})" if entry.get("reason") else ""
