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

"""``parlant lint``: lint rules for a Parlant agent repository, run by a coding agent.

The CLI judges nothing itself. It prints prompts: ``parlant lint`` tells the
coding agent how to run the lint, and ``parlant lint --section S --chapter N``
gives one subagent the rules to review the repository against.

Each section is a Markdown file in ``lint_rules/``, named after the section.
Its H1 is the section's title, and each H2 is one rule, written
``## ABC123 (SEVERITY): Title``. Chapters are not written anywhere: they are
the section's rules in file order, ``RULES_PER_CHAPTER`` at a time.

Rule IDs are stable across releases. ``lint_rules/RULE_IDS`` lists every ID
ever issued, and the tests hold the shipped rules to it.
"""

from __future__ import annotations

import math
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Sequence

import click

from parlant.bin.help import examples

RULES_DIR = Path(__file__).resolve().parent / "lint_rules"

RULES_PER_CHAPTER = 10

# Inside the repository's git directory, so recording a lint never shows up
# as a change to commit. The scaffolded pre-push hook reads the same file.
LINTED_COMMIT_FILE = "parlant-lint-passed"

_RULE_HEADING = re.compile(r"^## (?P<id>[A-Z]{3}\d{3}) \((?P<severity>[A-Z]+)\): (?P<title>\S.*)$")
_SECTION_NAME = re.compile(r"^[a-z0-9_]+$")
_FENCE = re.compile(r"^\s*(```|~~~)")


class LintRulesError(Exception):
    """A rule file or the ID registry is malformed."""


class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class Rule:
    id: str
    severity: Severity
    title: str
    body: str

    def as_markdown(self) -> str:
        return f"## {self.id} ({self.severity.value}): {self.title}\n\n{self.body}".rstrip()


@dataclass(frozen=True)
class Section:
    name: str
    title: str
    rules: Sequence[Rule]

    @property
    def chapter_count(self) -> int:
        return math.ceil(len(self.rules) / RULES_PER_CHAPTER)

    def chapter(self, number: int) -> Sequence[Rule]:
        """The rules of chapter ``number``, counting from 1."""
        if not 1 <= number <= self.chapter_count:
            raise ValueError(f"{self.name} has chapters 1 to {self.chapter_count}")
        start = (number - 1) * RULES_PER_CHAPTER
        return self.rules[start : start + RULES_PER_CHAPTER]


@dataclass(frozen=True)
class RuleIdRegistry:
    active: Sequence[str]
    retired: Sequence[str]


def _parse_section(path: Path) -> Section:
    name = path.stem
    if not _SECTION_NAME.match(name):
        raise LintRulesError(
            f"{path.name}: a section's file name must use only lowercase letters, "
            "digits and underscores, since it is typed as --section."
        )

    title: str | None = None
    rules: list[Rule] = []
    current: dict[str, str] | None = None
    body: list[str] = []
    in_fence = False

    def finish() -> None:
        if current is not None:
            rules.append(
                Rule(
                    id=current["id"],
                    severity=Severity(current["severity"]),
                    title=current["title"],
                    body="\n".join(body).strip() + "\n",
                )
            )

    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and line.startswith("# "):
            if title is not None or current is not None:
                raise LintRulesError(
                    f"{path.name}:{number}: only one H1 is allowed, before the first rule."
                )
            title = line[2:].strip()
            continue
        elif not in_fence and line.startswith("## "):
            match = _RULE_HEADING.match(line)
            if match is None:
                raise LintRulesError(
                    f"{path.name}:{number}: a rule heading must read "
                    "'## ABC123 (LOW|MEDIUM|HIGH): Title'."
                )
            if match["severity"] not in Severity.__members__:
                raise LintRulesError(
                    f"{path.name}:{number}: {match['id']} has severity "
                    f"{match['severity']}; use LOW, MEDIUM or HIGH."
                )
            if title is None:
                raise LintRulesError(f"{path.name}:{number}: the H1 title must come first.")
            finish()
            current = match.groupdict()
            body = []
            continue

        if current is not None:
            body.append(line)
        elif line.strip() and not in_fence and title is not None:
            # Text between the title and the first rule would belong to no
            # chapter, so no subagent would ever read it.
            raise LintRulesError(
                f"{path.name}:{number}: put text inside a rule; "
                "nothing may come between the H1 and the first rule."
            )

    finish()

    if title is None:
        raise LintRulesError(f"{path.name}: the file needs an H1 title.")
    if not rules:
        raise LintRulesError(f"{path.name}: the section has no rules.")

    return Section(name=name, title=title, rules=rules)


def load_sections(rules_dir: Path = RULES_DIR) -> list[Section]:
    """Every section in ``rules_dir``, in file-name order."""
    sections = [_parse_section(path) for path in sorted(rules_dir.glob("*.md"))]

    seen: dict[str, str] = {}
    for section in sections:
        for rule in section.rules:
            if rule.id in seen:
                raise LintRulesError(
                    f"{rule.id} is used in both {seen[rule.id]}.md and {section.name}.md."
                )
            seen[rule.id] = section.name

    return sections


def registered_rule_ids(rules_dir: Path = RULES_DIR) -> RuleIdRegistry:
    active: list[str] = []
    retired: list[str] = []

    for number, raw in enumerate((rules_dir / "RULE_IDS").read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) == 1:
            active.append(parts[0])
        elif len(parts) == 2 and parts[1] == "retired":
            retired.append(parts[0])
        else:
            raise LintRulesError(f"RULE_IDS:{number}: expected 'ID' or 'ID retired'.")

    return RuleIdRegistry(active=active, retired=retired)


def _command(section: Section, chapter: int) -> str:
    return f"parlant lint --section {section.name} --chapter {chapter}"


def overview_prompt(sections: Sequence[Section]) -> str:
    chapters = "\n".join(
        f"`{_command(section, chapter)}`"
        for section in sections
        for chapter in range(1, section.chapter_count + 1)
    )

    return f"""\
Here are the lint chapters you must run. Run each of these in a separate subagent/task, where \
each subagent first runs its section/chapter command, then reviews the repo for issues and \
reports its findings back to you.

After all subagents/tasks have run, look at their combined findings and address them. One \
iteration is one round of running the chapters and addressing what they found. Run a maximum \
of 2 iterations at a time - sometimes it's just impossible to solve everything at once.

Every finding has a severity:

- HIGH: always point these out to the user, and recommend fixing them immediately. In the \
first iteration, fix them yourself, without asking the user first.
- MEDIUM: per iteration, recommend fixing up to 3 of the MEDIUM issue types found, no more, \
even if 10 types were found. Fix them only with the user's consent.
- LOW: always ask the user before fixing these.

Whenever you communicate issues to the user, list them in brief bullet points, and use up to \
2 sentences per issue to describe what it is and its implications.

After 2 iterations, tell the user what was fixed in this lint stage, and flag only the HIGH \
findings that are still open. Let the rest be addressed quietly next time.

Then commit your fixes and run `parlant lint --done`. It records this commit as linted, so \
the repository's pre-push hook lets it through.

The chapters:
{chapters}
"""


def chapter_prompt(section: Section, chapter: int) -> str:
    rules = "\n\n".join(rule.as_markdown() for rule in section.chapter(chapter))

    return f"""\
# Lint: {section.title}, chapter {chapter} of {section.chapter_count}

Review this Parlant agent repository against the rules below, and only these rules. Read the \
agent's code (its agents, policies, observations, canned responses, tools and the rest), and \
check every rule against all of it. If you need to know how a Parlant construct behaves to \
judge a rule, run `parlant docs query` rather than relying on memory.

Do not change any files. Report your findings back to the agent that started you:

- For each finding: the rule ID and severity, the file and line, and up to 2 sentences on \
what the issue is and its implications.
- Group findings by rule ID, and say "no findings" for a rule that found none.
- Report only what these rules describe.

{rules}
"""


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def _record_linted_commit() -> None:
    git_dir = _git("rev-parse", "--absolute-git-dir")
    if git_dir.returncode != 0:
        raise click.ClickException(
            "This isn't a git repository. Run `parlant lint --done` from inside your "
            "Parlant project's folder."
        )

    head = _git("rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        raise click.ClickException(
            "This repository has no commits yet. Commit your work first, then run "
            "`parlant lint --done`."
        )

    commit = head.stdout.strip()
    (Path(git_dir.stdout.strip()) / LINTED_COMMIT_FILE).write_text(commit + "\n")

    click.echo(f"Recorded {commit[:12]} as linted. You can push it now.")
    if _git("status", "--porcelain").stdout.strip():
        click.echo(
            "Note: there are uncommitted changes. Only the commit is recorded; after "
            "committing them, lint and run `parlant lint --done` again."
        )


@click.command(
    "lint",
    short_help="Print the lint instructions for your coding agent.",
    epilog=examples(
        "parlant lint",
        "parlant lint --section behavior --chapter 1",
        "parlant lint --done",
    ),
)
@click.option("--section", help="The lint section to print (see `parlant lint`).")
@click.option("--chapter", type=int, help="The chapter of that section to print.")
@click.option(
    "--done",
    is_flag=True,
    help="Record the current commit as linted, so the pre-push hook lets it through.",
)
def lint(section: Optional[str], chapter: Optional[int], done: bool) -> None:
    """Lint a Parlant agent repository, by way of your coding agent.

    Run with no options for the instructions: which chapters to run, each in
    its own subagent, and how to handle what they find. Each subagent then
    runs `parlant lint --section SECTION --chapter CHAPTER` for its rules.

    After linting and committing the fixes, run `parlant lint --done`."""
    if done:
        if section is not None or chapter is not None:
            raise click.UsageError("--done takes no --section or --chapter.")
        _record_linted_commit()
        return

    sections = load_sections()

    if section is None and chapter is None:
        click.echo(overview_prompt(sections), nl=False)
        return

    if section is None or chapter is None:
        raise click.UsageError(
            "Give --section and --chapter together, as listed by `parlant lint`."
        )

    by_name = {s.name: s for s in sections}
    if section not in by_name:
        raise click.UsageError(
            f"There's no section '{section}'. The sections are: "
            + ", ".join(by_name)
            + ". Run `parlant lint` for the full list of commands."
        )

    chosen = by_name[section]
    if not 1 <= chapter <= chosen.chapter_count:
        valid = ", ".join(f"--chapter {n}" for n in range(1, chosen.chapter_count + 1))
        raise click.UsageError(f"Section '{section}' has: {valid}.")

    click.echo(chapter_prompt(chosen, chapter), nl=False)
