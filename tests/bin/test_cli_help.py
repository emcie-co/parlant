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

"""`--help` is the documentation most people actually read, and the only one a
coding agent can discover on its own. These checks walk the real command tree,
so a command added later is held to the same standard without anyone
remembering to come back here.

They also resolve every example we publish — in `docs/parlant-cli.md` and in the
`AGENTS.md` we scaffold — against that same tree, so a renamed command or
dropped flag breaks a test rather than someone's copy-paste.
"""

import re
import shlex
from pathlib import Path
from typing import Iterator, Optional

import click
import pytest

from parlant.bin.cli import cli

_REPOSITORY_ROOT = Path(__file__).parents[2]
_CLI_DOC = _REPOSITORY_ROOT / "docs" / "parlant-cli.md"


def _walk(command: click.Command, path: list[str]) -> Iterator[tuple[str, click.Command]]:
    yield " ".join(path), command

    if isinstance(command, click.Group):
        for name, child in sorted(command.commands.items()):
            yield from _walk(child, path + [name])


_EVERY_COMMAND = list(_walk(cli, ["parlant"]))
_LEAF_COMMANDS = [(name, c) for name, c in _EVERY_COMMAND if not isinstance(c, click.Group)]


def _help_text(command: click.Command) -> str:
    return f"{command.help or ''}\n{command.epilog or ''}"


def _description(command: click.Command) -> str:
    """The prose part only.

    Deliberately excludes the epilog: an example shows an argument's shape
    without saying what it is or where to get one, and showing it is not the
    same as explaining it.
    """
    return command.help or ""


def _arguments(command: click.Command) -> list[click.Argument]:
    return [p for p in command.params if isinstance(p, click.Argument)]


def _options(command: click.Command) -> list[click.Option]:
    return [p for p in command.params if isinstance(p, click.Option)]


@pytest.mark.parametrize(
    "name,command", _EVERY_COMMAND, ids=[n.replace(" ", "_") for n, _ in _EVERY_COMMAND]
)
def test_that_every_command_explains_what_it_does(name: str, command: click.Command) -> None:
    assert (command.help or "").strip(), f"`{name}` has no help text"


@pytest.mark.parametrize(
    "name,command", _EVERY_COMMAND, ids=[n.replace(" ", "_") for n, _ in _EVERY_COMMAND]
)
def test_that_every_option_says_what_it_is_for(name: str, command: click.Command) -> None:
    undocumented = [max(o.opts, key=len) for o in _options(command) if not (o.help or "").strip()]

    assert not undocumented, f"`{name}` has options with no help: {', '.join(undocumented)}"


@pytest.mark.parametrize(
    "name,command", _LEAF_COMMANDS, ids=[n.replace(" ", "_") for n, _ in _LEAF_COMMANDS]
)
def test_that_every_positional_argument_is_explained_in_its_commands_help(
    name: str, command: click.Command
) -> None:
    """Click prints arguments in the usage line and never describes them.

    So an argument the help text does not mention reaches the reader as a bare
    metavar — `parlant dashboard trace view TRACE_ID` tells you a trace id goes
    there, but not which trace ids exist or where to find one. Naming it in the
    help is the only place that can be said.
    """
    text = _description(command)
    unexplained = [
        argument.name.upper()
        for argument in _arguments(command)
        if argument.name and argument.name.upper() not in text.upper()
    ]

    assert not unexplained, (
        f"`{name}` takes {', '.join(unexplained)} but its help never mentions it"
    )


@pytest.mark.parametrize(
    "name,command", _LEAF_COMMANDS, ids=[n.replace(" ", "_") for n, _ in _LEAF_COMMANDS]
)
def test_that_every_leaf_command_shows_a_usage_example(name: str, command: click.Command) -> None:
    epilog = command.epilog or ""

    assert f"parlant {name.removeprefix('parlant ')}" in epilog, (
        f"`{name}` shows no worked example; put one in the command's epilog"
    )


def _invocations(text: str) -> list[str]:
    """Every `parlant ...` command line in a markdown document's code blocks."""
    invocations: list[str] = []

    for block in re.findall(r"```[a-z]*\n(.*?)```", text, flags=re.DOTALL):
        # Join lines a shell would join, so a wrapped example is read whole.
        joined = re.sub(r"\\\n\s*", " ", block)

        for line in joined.splitlines():
            stripped = line.strip().removeprefix("$ ").strip()

            if stripped.startswith("parlant ") or stripped == "parlant":
                invocations.append(stripped)

    return invocations


def _resolution_failure(invocation: str) -> Optional[str]:
    """Why this command line does not match the real CLI, or None if it does."""
    try:
        tokens = shlex.split(invocation, comments=True)
    except ValueError as error:
        return f"could not be parsed as a shell command ({error})"

    command: click.Command = cli
    index = 1

    while index < len(tokens) and isinstance(command, click.Group):
        if tokens[index] not in command.commands:
            break
        command = command.commands[tokens[index]]
        index += 1

    if isinstance(command, click.Group) and index < len(tokens):
        candidate = tokens[index]
        if not candidate.startswith("-"):
            return f"`{candidate}` is not a command of `{' '.join(tokens[:index])}`"

    known = {opt for option in _options(command) for opt in option.opts + option.secondary_opts}

    for token in tokens[index:]:
        if not token.startswith("--"):
            continue
        flag = token.split("=", 1)[0]
        if flag not in known and flag != "--help":
            return f"`{flag}` is not an option of `{' '.join(tokens[:index])}`"

    return None


def test_that_every_cli_example_in_the_docs_resolves_to_a_real_command() -> None:
    assert _CLI_DOC.exists(), f"{_CLI_DOC.name} is missing"

    invocations = _invocations(_CLI_DOC.read_text())

    assert invocations, f"{_CLI_DOC.name} shows no CLI examples at all"

    failures = [
        f"{invocation}\n    -> {reason}"
        for invocation in invocations
        if (reason := _resolution_failure(invocation))
    ]

    assert not failures, (
        "docs/parlant-cli.md publishes examples that do not work:\n  " + "\n  ".join(failures)
    )


def test_that_every_cli_example_in_the_scaffolded_agents_file_resolves_to_a_real_command() -> None:
    """The scaffolded AGENTS.md tells a coding agent to consult `parlant docs`.

    Those examples are the first thing such an agent runs, so a flag that no
    longer exists costs it a failed command and some trust. Same resolver as the
    guide, against the same tree.
    """
    from parlant.bin.scaffold import Scaffold, agents_md

    for scaffold in Scaffold:
        invocations = _invocations(agents_md("my-agent", scaffold))

        assert invocations, f"the {scaffold.value} AGENTS.md shows no CLI examples"

        failures = [
            f"{invocation}\n    -> {reason}"
            for invocation in invocations
            if (reason := _resolution_failure(invocation))
        ]

        assert not failures, (
            f"the {scaffold.value} AGENTS.md publishes examples that do not work:\n  "
            + "\n  ".join(failures)
        )
