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

"""The ``parlant`` command line.

Grouped as ``parlant <group> <command>`` so later groups (for example, the
dashboard) slot in without reshaping the surface.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Optional

import click
from platformdirs import user_cache_dir
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt
from rich.table import Table
from rich.text import Text

from parlant.bin.dashboard import dashboard
from parlant.bin.dashboard.commands._shared import runtime_for
from parlant.bin.dashboard.commands.project_connection import connect
from parlant.bin.dashboard.commands.project_creation import create
from parlant.bin.dashboard.errors import DashboardError
from parlant.bin.dashboard.output import emit_json, json_mode, json_option, reset_json_mode
from parlant.bin.docs import docs
from parlant.bin.scaffold import (
    PackageManager,
    Scaffold,
    detect_package_managers,
    install_command,
    project_files,
    write_project,
)
from parlant.bin.help import examples

# The widget source ships inside the package (see the force-include in
# pyproject.toml). In a source checkout it lives at the repo's widget/ instead.
_PACKAGED_WIDGET_DIR = Path(__file__).resolve().parent.parent / "_widget"
_CHECKOUT_WIDGET_DIR = Path(__file__).resolve().parents[3] / "widget"

VERSION_STAMP_FILE = ".parlant-widget-version"

PLAYGROUND_DIRNAME = "widget-playground"

DEFAULT_PLAYGROUND_PORT = 5173
DEFAULT_AGENT_PORT = 8810
SIGNED_IN_AGENT_PORT = 8811

# Copied into an app; the harness around it stays behind.
VENDORED_WIDGET_DIRNAME = "parlant-chat"


def parlant_version() -> str:
    try:
        return metadata.version("parlant")
    except metadata.PackageNotFoundError:
        return "0.0.0+unknown"


def default_playground_dir() -> Path:
    """Where the playground is copied to when no directory is given.

    The OS cache directory, because the playground's real cost is its ~127MB
    npm install: keeping it there means one install per machine rather than
    one per project, and nothing is left behind in whatever directory the
    command was run from. It cannot run in place — npm install would write
    into site-packages, which may be read-only and is wiped on upgrade.
    """
    return Path(user_cache_dir("parlant")) / PLAYGROUND_DIRNAME


def widget_source_dir() -> Path:
    """The bundled widget source, from the installed package or a checkout."""
    for candidate in (_PACKAGED_WIDGET_DIR, _CHECKOUT_WIDGET_DIR):
        if (candidate / VENDORED_WIDGET_DIRNAME).is_dir():
            return candidate

    raise click.ClickException(
        "The bundled widget source is missing from this Parlant installation. "
        "Reinstall Parlant, or run this command from a Parlant checkout."
    )


def _fail(message: str) -> "click.ClickException":
    return click.ClickException(message)


def _port_is_free(port: int) -> bool:
    # Checked on both stacks: a listener bound to [::1] leaves 127.0.0.1
    # bindable, so an IPv4-only probe would call an occupied port free and
    # the dev server would then fail (or quietly move) on startup.
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((address, port))
        except OSError as error:
            import errno

            if error.errno in (errno.EADDRINUSE, errno.EACCES):
                return False
            # This host may not support the family at all; that is not a
            # port conflict.
            continue
    return True


def _require_free_port(port: int, purpose: str) -> None:
    if not _port_is_free(port):
        raise _fail(
            f"Port {port} is already in use, and the {purpose} needs it. "
            "Stop whatever is listening there, or pass a different port."
        )


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(
            "node_modules", "dist", "parlant-data", ".env", "*.tsbuildinfo"
        ),
    )


@click.group()
@click.version_option(parlant_version(), "-v", "--version", prog_name="parlant")
def cli() -> None:
    """Parlant's command line."""
    reset_json_mode()


@cli.group()
def widget() -> None:
    """Work with the Parlant chat widget."""


cli.add_command(dashboard)
cli.add_command(docs)


@widget.command(
    "init",
    epilog=examples(
        "parlant widget init",
        "parlant widget init ./web/chat",
        "parlant widget init ./web/chat --force",
    ),
)
@click.argument(
    "destination",
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite the destination if it already exists.",
)
def widget_init(destination: Optional[Path], force: bool) -> None:
    """Copy the chat widget into your app.

    DESTINATION is the folder to copy it into, and defaults to `parlant-chat`
    in the current directory.

    The widget is yours once copied: restyle it, swap the transport, add your
    own tool widgets. This is a file copy, so it needs no node toolchain.
    """
    source = widget_source_dir() / VENDORED_WIDGET_DIRNAME
    target = destination or Path(VENDORED_WIDGET_DIRNAME)

    if target.exists() and any(target.iterdir()) and not force:
        raise _fail(
            f"'{target}' already exists and is not empty. "
            "Pass --force to overwrite it, or choose another destination."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    _copy_tree(source, target)

    version = parlant_version()
    (target / VERSION_STAMP_FILE).write_text(
        f"{version}\n"
        "# The Parlant version this widget was copied from. Its transport speaks\n"
        "# that version's HTTP API; re-run `parlant widget init --force` after a\n"
        "# major server upgrade.\n"
    )

    click.echo(f"Copied the Parlant chat widget to '{target}' (Parlant {version}).")
    click.echo("")
    click.echo("Next: render it in your app.")
    click.echo("")
    click.echo(f'    import {{ ChatWidget }} from "./{target.name}";')
    click.echo("")
    click.echo('    <ChatWidget serverUrl="http://localhost:8800" agentId="your-agent-id" />')


# --- parlant init ---------------------------------------------------------


def _project_name_from(destination: Path) -> str:
    name = destination.resolve().name or "parlant-agent"
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower() or "parlant-agent"


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def _confirm_current_directory(console: Console, target: Path) -> None:
    """Check we're in the right place before filling the working directory.

    Only when no destination was given: an explicit one is already the user
    saying where. Skipped when there's nobody to answer, so scripts and CI
    don't hang on a prompt.
    """
    if not _is_interactive():
        return

    console.print()

    if not Confirm.ask(
        f"Create the project here, in [bold]{target.resolve()}[/bold]?",
        default=True,
        console=console,
    ):
        console.print("[dim]Nothing was written.[/dim]")
        raise click.Abort()


def _choose_package_manager(console: Console) -> PackageManager:
    """The interview: what each option is, and whether this machine has it."""
    detected = detect_package_managers()

    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    table.add_column(justify="right", style="bold cyan", no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(no_wrap=True)
    table.add_column(style="dim")

    for index, detection in enumerate(detected, start=1):
        status = (
            Text("● installed", style="green")
            if detection.installed
            else Text("● not installed", style="red")
        )
        table.add_row(f"{index}.", Text(detection.label, style="bold"), status, detection.note)

    console.print()
    console.print("[bold]How should this project manage its dependencies?[/bold]")
    console.print(table)

    choice = IntPrompt.ask(
        "Choose",
        choices=[str(i) for i in range(1, len(detected) + 1)],
        default=1,
        show_choices=False,
        console=console,
    )

    selection = detected[choice - 1]

    if not selection.installed:
        console.print(
            f"[yellow]Note:[/yellow] {selection.label} isn't installed yet — "
            "the project will be set up for it, and you'll need it to run."
        )

    return selection.manager


def _init_git_repository(destination: Path, console: Console) -> None:
    if shutil.which("git") is None:
        console.print("[dim]git not found; skipping repository setup.[/dim]")
        return

    existing = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=destination,
        capture_output=True,
        text=True,
    )

    if existing.returncode == 0 and existing.stdout.strip() == "true":
        # Nesting a repository inside another is a mess to undo.
        console.print("[dim]Already inside a git repository; skipping git init.[/dim]")
        return

    subprocess.run(["git", "init", "--quiet"], cwd=destination, check=True)
    console.print("[dim]Initialized a git repository.[/dim]")


@cli.command(
    "init",
    epilog=examples(
        "parlant init",
        "parlant init my-agent",
        "parlant init my-agent --scaffold retail-store --package-manager uv",
        "parlant init my-agent --scaffold lead-gen --package-manager uv",
        "parlant init my-agent --dashboard --yes",
    ),
)
@click.argument(
    "destination",
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--package-manager",
    type=click.Choice([m.value for m in PackageManager]),
    default=None,
    help="Skip the interview and use this package manager.",
)
@click.option(
    "--scaffold",
    type=click.Choice([s.value for s in Scaffold]),
    default=Scaffold.BASIC.value,
    show_default=True,
    help=(
        "What to fill the project with: an empty structure (basic), or a worked "
        "example: a support agent (retail-store) or a website lead-generation "
        "agent (lead-gen)."
    ),
)
@click.option("--force", is_flag=True, help="Write into a non-empty directory.")
@click.option("--no-git", is_flag=True, help="Don't initialize a git repository.")
@click.option(
    "--dashboard",
    "with_dashboard",
    is_flag=True,
    help="Create it as a Local project on the Parlant dashboard first, and connect it.",
)
@click.option("--name", help="The dashboard project's name (default: the folder's name).")
@click.option("--workspace", help="The workspace to create the dashboard project in.")
@click.option("--yes", is_flag=True, help="Create the dashboard project without asking.")
@json_option
@click.pass_context
def init(
    ctx: click.Context,
    destination: Optional[Path],
    package_manager: Optional[str],
    scaffold: str,
    force: bool,
    no_git: bool,
    with_dashboard: bool,
    name: Optional[str],
    workspace: Optional[str],
    yes: bool,
) -> None:
    """Create a new Parlant agent project.

    DESTINATION is the folder to scaffold into, and defaults to the current
    directory (which it asks about first, since that is the one you cannot
    undo by deleting a folder). An existing folder is fine; the command
    refuses only on the individual files it would overwrite.

    With --dashboard, the project is created on the dashboard first, so nothing
    is written unless the dashboard accepts it; its token then goes in .env."""
    # With --json, stdout carries only the result.
    console = Console(stderr=json_mode())
    target = destination or Path(".")
    chosen = Scaffold(scaffold)

    if json_mode() and package_manager is None:
        raise DashboardError("--json needs --package-manager: it never prompts.")
    if json_mode() and with_dashboard and not yes:
        raise DashboardError("--json needs --yes: it never prompts.")
    if (name or workspace) and not with_dashboard:
        raise _fail("--name and --workspace apply only with --dashboard.")

    if destination is None and not json_mode():
        _confirm_current_directory(console, target)

    manager = (
        PackageManager(package_manager)
        if package_manager is not None
        else _choose_package_manager(console)
    )

    project_name = _project_name_from(target)
    version = parlant_version()

    # Refuse on files we would overwrite, rather than on a non-empty
    # directory: scaffolding into a freshly created git repository, or
    # alongside a README, is normal and shouldn't be blocked.
    if not force:
        clashes = [str(p) for p in project_files(target, manager, chosen) if p.exists()]

        if clashes:
            listed = ", ".join(sorted(clashes)[:4])
            more = "" if len(clashes) <= 4 else f" (and {len(clashes) - 4} more)"
            raise DashboardError(
                f"These files already exist: {listed}{more}. "
                "Pass --force to overwrite them, or choose another directory.",
                code="FILES_EXIST",
            )

    project_id: Optional[str] = None
    if with_dashboard:
        # The dashboard admits the project before anything is written.
        runtime = runtime_for(ctx)
        if workspace:
            runtime.workspace = workspace
        created = create(runtime, name=name or project_name, kind="local", yes=yes)
        if created is None:
            return
        project_id = str(created["id"])

    target.mkdir(parents=True, exist_ok=True)

    with console.status("[bold green]Creating your project...", spinner="dots"):
        files = write_project(target, project_name, manager, version, chosen)

    console.print(f"[green]✔[/green] Wrote {len(files)} files to [bold]{target}[/bold]")

    if not no_git:
        _init_git_repository(target, console)

    env_file: Optional[str] = None
    if project_id is not None:
        env_file = connect(runtime_for(ctx), project_id, target)["envFile"]
        console.print(
            f"[green]✔[/green] Connected to the dashboard project [bold]{project_id}[/bold]; "
            "its token is in .env"
        )

    if json_mode():
        emit_json(
            {
                "dir": str(target),
                "files": [str(path.relative_to(target)) for path in files],
                **({"projectId": project_id, "envFile": env_file} if project_id else {}),
            }
        )
        return

    if project_id is not None:
        _print_dashboard_next_steps(console, target, manager, project_name, project_id)
        return

    where = "" if target == Path(".") else f"cd {target}\n"
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]{where}{install_command(manager, project_name)}[/bold]\n"
                "[dim]install dependencies[/dim]\n\n"
                "[bold]cp .env.example .env[/bold]\n"
                "[dim]then set OPENROUTER_API_KEY[/dim]\n\n"
                "[bold]./run.sh[/bold]  [dim](run.cmd on Windows)[/dim]\n"
                "[dim]start the agent, then open http://localhost:8800/chat[/dim]"
            ),
            title="[bold green]Next[/bold green]",
            border_style="green",
        )
    )

    _print_agents_note(console)


def _print_agents_note(console: Console) -> None:
    """Addressed to coding agents, because they are often the ones running this.

    An agent that scaffolds a project and then writes code from its own priors
    is what the scaffolded AGENTS.md exists to prevent, and this output is the
    only place it reliably reads.
    """
    console.print()
    console.print("For agents: please first read AGENTS.md in the created project.")
    console.print()


def _print_dashboard_next_steps(
    console: Console,
    target: Path,
    manager: PackageManager,
    project_name: str,
    project_id: str,
) -> None:
    where = "" if target == Path(".") else f"cd {target}\n"
    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]{where}{install_command(manager, project_name)}[/bold]\n"
                "[dim]install dependencies, then set OPENROUTER_API_KEY in .env[/dim]\n\n"
                "[bold]./run.sh[/bold]  [dim](run.cmd on Windows)[/dim]\n"
                "[dim]start the agent; it connects to the dashboard on its own[/dim]\n\n"
                f"[bold]parlant dashboard --project {project_id} project check[/bold]\n"
                "[dim]see that the dashboard reaches it, then[/dim]\n"
                f"[bold]parlant dashboard --project {project_id} project confirm[/bold]\n"
                "[dim]to complete the project's creation[/dim]"
            ),
            title="[bold green]Next[/bold green]",
            border_style="green",
        )
    )
    _print_agents_note(console)


def _preflight_npm() -> str:
    npm = shutil.which("npm")
    if npm is None:
        raise _fail(
            "npm was not found on your PATH, and the playground needs it to "
            "install the demo's dependencies. Install Node.js "
            "(https://nodejs.org) and try again.\n"
            "To vendor the widget without any node toolchain, run "
            "`parlant widget init` instead."
        )
    return npm


def _stream_process(name: str, process: subprocess.Popen[str]) -> None:
    assert process.stdout is not None
    for line in process.stdout:
        click.echo(f"[{name}] {line.rstrip()}")


@widget.command(
    "playground",
    epilog=examples(
        "parlant widget playground",
        "parlant widget playground --port 8000",
        "parlant widget playground --no-agent",
    ),
)
@click.option(
    "--dir",
    "directory",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="Where to copy the playground (default: your OS cache directory).",
)
@click.option(
    "--port",
    default=DEFAULT_PLAYGROUND_PORT,
    show_default=True,
    help="Port for the playground's dev server.",
)
@click.option(
    "--no-agent",
    is_flag=True,
    help="Don't start the bundled sample agent; point the playground at your own server.",
)
@click.option(
    "--signed-in",
    is_flag=True,
    help="Run the sample agent with customer-JWT auth and print a token to paste in.",
)
def widget_playground(
    directory: Optional[Path],
    port: int,
    no_agent: bool,
    signed_in: bool,
) -> None:
    """Run the widget playground against a sample agent.

    Copies the demo out of the installed package, installs its dependencies,
    and runs it. With the sample agent, this is a complete working chat.
    """
    # Preflight everything before copying or installing, so a missing
    # prerequisite costs the user nothing but the message.
    npm = _preflight_npm()

    if not no_agent and not os.environ.get("OPENAI_API_KEY"):
        raise _fail(
            "OPENAI_API_KEY is not set, and the bundled sample agent uses "
            "OpenAI for both chat and vision.\n"
            "Set it, or pass --no-agent to run the playground against your "
            "own server."
        )

    _require_free_port(port, "playground's dev server")

    agent_port = SIGNED_IN_AGENT_PORT if signed_in else DEFAULT_AGENT_PORT
    if not no_agent:
        _require_free_port(agent_port, "sample agent")

    source = widget_source_dir()
    directory = directory or default_playground_dir()

    click.echo(f"Setting up the playground in '{directory}'...")
    directory.parent.mkdir(parents=True, exist_ok=True)
    _copy_tree(source, directory)

    if (directory / "node_modules").is_dir():
        click.echo("Dependencies are already installed; skipping npm install.")
    else:
        click.echo("Installing dependencies (npm install)...")
        install = subprocess.run([npm, "install"], cwd=directory)
        if install.returncode != 0:
            raise _fail("npm install failed. See the output above.")

    processes: list[tuple[str, subprocess.Popen[str]]] = []

    # SIGTERM would otherwise kill this supervisor outright, leaving the agent
    # and dev server holding their ports. Turning it into KeyboardInterrupt
    # routes it through the same cleanup as Ctrl+C.
    def _terminate(signal_number: int, frame: object) -> None:
        raise KeyboardInterrupt

    previous_sigterm = signal.signal(signal.SIGTERM, _terminate)

    try:
        if not no_agent:
            script = "signed_in_agent.py" if signed_in else "agent.py"
            click.echo(f"Starting the sample agent on port {agent_port}...")

            agent_process = subprocess.Popen(
                [sys.executable, script],
                # The signed-in variant imports its tool from agent.py, so the
                # sample-agent directory has to be the working directory.
                cwd=directory / "sample-agent",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            processes.append(("agent", agent_process))

            _wait_for_agent(agent_process, agent_port)

            if signed_in:
                _print_signed_in_token(directory)

        click.echo(f"Starting the playground on http://localhost:{port} ...")
        vite_process = subprocess.Popen(
            # --strictPort: without it Vite quietly moves to the next free
            # port, and the URL printed above would be wrong.
            [npm, "run", "dev", "--", "--port", str(port), "--strictPort"],
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        processes.append(("playground", vite_process))

        click.echo("")
        click.echo("Press Ctrl+C to stop.")
        click.echo("")

        _stream_process("playground", vite_process)
    except KeyboardInterrupt:
        click.echo("\nStopping...")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

        for _, process in reversed(processes):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for _, process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def _wait_for_agent(process: subprocess.Popen[str], port: int, timeout: float = 180.0) -> None:
    """Block until the sample agent answers /healthz, or fail loudly."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise _fail("The sample agent exited before it finished starting. Its output is above.")

        try:
            with urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=3):
                click.echo("The sample agent is ready.")
                return
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)

    raise _fail(f"The sample agent did not become ready within {int(timeout)} seconds.")


def _print_signed_in_token(directory: Path) -> None:
    minted = subprocess.run(
        [sys.executable, "mint_token.py"],
        cwd=directory / "sample-agent",
        capture_output=True,
        text=True,
    )

    if minted.returncode != 0:
        click.echo("Could not mint a demo token; run mint_token.py yourself.")
        return

    click.echo("")
    click.echo("Signed-in mode. In the playground, pick 'Signed in', set the server")
    click.echo(f"URL to http://localhost:{SIGNED_IN_AGENT_PORT}, and paste this token:")
    click.echo("")
    click.echo(f"    {minted.stdout.strip()}")
    click.echo("")


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
