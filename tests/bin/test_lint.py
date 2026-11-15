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

import re
import subprocess
from pathlib import Path

from click.testing import CliRunner
import pytest

from parlant.bin.cli import cli
from parlant.bin.lint import (
    RULES_DIR,
    LintRulesError,
    chapter_prompt,
    load_sections,
    overview_prompt,
    registered_rule_ids,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _section_file(directory: Path, name: str, title: str, rule_count: int) -> Path:
    prefix = name[:3].upper()
    rules = "\n\n".join(
        f"## {prefix}{i:03d} (LOW): Rule number {i}\n\nThe body of rule {i}."
        for i in range(1, rule_count + 1)
    )
    path = directory / f"{name}.md"
    path.write_text(f"# {title}\n\n{rules}\n")
    return path


def test_that_lint_lists_one_command_per_section_and_chapter(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["lint"])

    assert result.exit_code == 0, result.output

    commands = re.findall(r"`(parlant lint --section \S+ --chapter \d+)`", result.output)
    expected = [
        f"parlant lint --section {section.name} --chapter {chapter}"
        for section in load_sections(RULES_DIR)
        for chapter in range(1, section.chapter_count + 1)
    ]
    assert commands == expected
    assert "parlant lint --section behavior --chapter 1" in commands
    assert "parlant lint --section wording_and_tone_of_voice --chapter 1" in commands


def test_that_a_section_with_fifteen_rules_has_two_chapters(tmp_path: Path) -> None:
    _section_file(tmp_path, "custom", "Custom rules", 15)

    (section,) = load_sections(tmp_path)

    assert section.chapter_count == 2
    assert "parlant lint --section custom --chapter 1" in overview_prompt([section])
    assert "parlant lint --section custom --chapter 2" in overview_prompt([section])
    assert "--chapter 3" not in overview_prompt([section])


def test_that_a_chapter_holds_at_most_ten_rules(tmp_path: Path) -> None:
    _section_file(tmp_path, "custom", "Custom rules", 15)

    (section,) = load_sections(tmp_path)

    assert [rule.id for rule in section.chapter(1)] == [f"CUS{i:03d}" for i in range(1, 11)]
    assert [rule.id for rule in section.chapter(2)] == [f"CUS{i:03d}" for i in range(11, 16)]


def test_that_a_section_is_titled_by_its_h1(tmp_path: Path) -> None:
    _section_file(tmp_path, "custom", "A Custom Title", 1)

    (section,) = load_sections(tmp_path)

    assert section.name == "custom"
    assert section.title == "A Custom Title"
    assert "A Custom Title" in chapter_prompt(section, 1)


def test_that_the_overview_states_how_to_handle_each_severity(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["lint"])

    assert result.exit_code == 0, result.output
    output = result.output

    assert "separate subagent" in output
    assert "maximum of 2 iterations" in output
    # HIGH: always pointed out, and fixed in the first iteration without asking.
    assert re.search(r"HIGH.*first iteration.*without asking", output, re.S)
    # MEDIUM: at most 3 issue types per iteration, with consent.
    assert re.search(r"MEDIUM.*up to 3.*consent", output, re.S)
    # LOW: always ask first.
    assert re.search(r"LOW.*always ask the user", output, re.S)
    assert "up to 2 sentences" in output
    assert "brief bullet points" in output
    # Once linted and committed, the push goes through.
    assert "parlant lint --done" in output


def test_that_a_chapter_prints_its_rules_with_ids_and_severities(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["lint", "--section", "behavior", "--chapter", "1"])

    assert result.exit_code == 0, result.output
    assert "BHV001" in result.output
    assert "HIGH" in result.output
    assert "BHV002" in result.output
    assert "MEDIUM" in result.output
    # The subagent reports back; it does not fix.
    assert "Do not change any files" in result.output
    # Another section's rules stay out of this chapter.
    assert "WTV001" not in result.output


def test_that_lint_refuses_an_unknown_section_and_names_the_valid_ones(
    runner: CliRunner,
) -> None:
    result = runner.invoke(cli, ["lint", "--section", "nonsense", "--chapter", "1"])

    assert result.exit_code != 0
    assert "behavior" in result.output
    assert "wording_and_tone_of_voice" in result.output


def test_that_lint_refuses_a_chapter_out_of_range(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["lint", "--section", "behavior", "--chapter", "99"])

    assert result.exit_code != 0
    assert "--chapter 1" in result.output


def test_that_lint_refuses_a_section_without_a_chapter(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["lint", "--section", "behavior"])

    assert result.exit_code != 0
    assert "--chapter" in result.output


def test_that_a_rule_without_a_valid_severity_is_refused(tmp_path: Path) -> None:
    (tmp_path / "custom.md").write_text("# Custom\n\n## CUS001 (URGENT): Something\n\nBody.\n")

    with pytest.raises(LintRulesError, match="custom.md"):
        load_sections(tmp_path)


def test_that_every_shipped_rule_id_is_unique_and_registered() -> None:
    ids = [rule.id for section in load_sections(RULES_DIR) for rule in section.rules]

    assert len(ids) == len(set(ids))
    registry = registered_rule_ids(RULES_DIR)
    assert set(ids) <= set(registry.active)


def test_that_no_registered_rule_id_disappears_without_being_retired() -> None:
    shipped = {rule.id for section in load_sections(RULES_DIR) for rule in section.rules}
    registry = registered_rule_ids(RULES_DIR)

    # An ID in the registry is either shipped or explicitly retired, never silently gone.
    assert set(registry.active) == shipped
    assert not set(registry.active) & set(registry.retired)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_that_the_pre_push_hook_blocks_a_push_until_lint_is_recorded_for_the_commit(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--quiet", "--bare", str(remote)], check=True)

    project = tmp_path / "bot"
    result = runner.invoke(cli, ["init", str(project), "--package-manager", "uv"])
    assert result.exit_code == 0, result.output

    _git(project, "add", "-A")
    assert _git(project, "commit", "--quiet", "-m", "init").returncode == 0
    _git(project, "remote", "add", "origin", str(remote))

    blocked = _git(project, "push", "--quiet", "origin", "HEAD:main")
    assert blocked.returncode != 0
    assert "parlant lint" in blocked.stderr

    monkeypatch.chdir(project)
    done = runner.invoke(cli, ["lint", "--done"])
    assert done.exit_code == 0, done.output

    allowed = _git(project, "push", "--quiet", "origin", "HEAD:main")
    assert allowed.returncode == 0, allowed.stderr

    # A new commit needs a new lint.
    (project / "notes.txt").write_text("change\n")
    _git(project, "add", "-A")
    _git(project, "commit", "--quiet", "-m", "change")
    assert _git(project, "push", "--quiet", "origin", "HEAD:main").returncode != 0


def test_that_lint_done_outside_a_git_repository_says_what_to_do(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli, ["lint", "--done"])

    assert result.exit_code != 0
    assert "git repository" in result.output
