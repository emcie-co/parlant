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

import socket
from pathlib import Path

from click.testing import CliRunner
import pytest

from parlant.bin.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_that_the_cli_exposes_a_widget_group_and_reports_its_version(
    runner: CliRunner,
) -> None:
    help_result = runner.invoke(cli, ["--help"])
    assert help_result.exit_code == 0
    assert "widget" in help_result.output

    version_result = runner.invoke(cli, ["--version"])
    assert version_result.exit_code == 0
    assert any(character.isdigit() for character in version_result.output)


def test_that_widget_init_copies_the_widget_source_into_the_destination(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["widget", "init"])

        assert result.exit_code == 0, result.output

        widget_dir = Path("parlant-chat")
        assert (widget_dir / "index.ts").is_file()
        assert (widget_dir / "styles.css").is_file()
        assert (widget_dir / "lib" / "transport.ts").is_file()
        assert (widget_dir / "components").is_dir()

        # The harness is not part of what an app vendors.
        assert not (widget_dir / "node_modules").exists()
        assert not (widget_dir / "example").exists()
        assert not (widget_dir / "package.json").exists()


def test_that_widget_init_accepts_an_explicit_destination(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["widget", "init", "src/vendor/chat"])

        assert result.exit_code == 0, result.output
        assert Path("src/vendor/chat/index.ts").is_file()


def test_that_widget_init_records_the_parlant_version_it_came_from(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        assert runner.invoke(cli, ["widget", "init"]).exit_code == 0

        stamp = Path("parlant-chat") / ".parlant-widget-version"
        assert stamp.is_file()
        assert any(character.isdigit() for character in stamp.read_text())


def test_that_widget_init_refuses_to_overwrite_an_existing_directory_without_force(
    runner: CliRunner,
) -> None:
    with runner.isolated_filesystem():
        destination = Path("parlant-chat")
        destination.mkdir()
        (destination / "my-edit.ts").write_text("// customized by the app")

        result = runner.invoke(cli, ["widget", "init"])

        assert result.exit_code != 0
        assert "--force" in result.output
        # The user's own file must survive a refused run.
        assert (destination / "my-edit.ts").is_file()

        forced = runner.invoke(cli, ["widget", "init", "--force"])

        assert forced.exit_code == 0, forced.output
        assert (destination / "index.ts").is_file()


def test_that_widget_init_works_without_any_node_toolchain(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Vendoring the widget is a file copy: it must not require node or npm.
    monkeypatch.setenv("PATH", "")

    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["widget", "init"])

        assert result.exit_code == 0, result.output
        assert Path("parlant-chat/index.ts").is_file()


def test_that_the_playground_fails_clearly_when_npm_is_missing(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "")

    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["widget", "playground"])

        assert result.exit_code != 0
        assert "npm" in result.output
        # A missing tool is a user-facing condition, not a crash.
        assert result.exception is None or isinstance(result.exception, SystemExit)


def test_that_the_playground_reports_a_missing_model_api_key_before_starting_anything(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["widget", "playground"])

        assert result.exit_code != 0
        assert "OPENAI_API_KEY" in result.output
        # Nothing should have been copied or installed before the check failed.
        assert not Path("parlant-widget").exists()


def test_that_the_playground_fails_clearly_when_a_required_port_is_in_use(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    taken_port = blocker.getsockname()[1]

    try:
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli, ["widget", "playground", "--port", str(taken_port), "--no-agent"]
            )

            assert result.exit_code != 0
            assert str(taken_port) in result.output
            assert "in use" in result.output.lower()
    finally:
        blocker.close()


def test_that_the_playground_defaults_to_the_user_cache_directory(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The playground's cost is its ~127MB npm install, so it belongs in the
    # OS cache directory: reused across projects and reboots, and never
    # littering the directory the user happened to run from.
    from parlant.bin.cli import default_playground_dir

    resolved = default_playground_dir()

    assert resolved.is_absolute()
    assert "widget-playground" in resolved.name
    assert Path.cwd() not in resolved.parents

    monkeypatch.setenv("PATH", "")

    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["widget", "playground"])

        # It fails on the missing toolchain, not before — and it left nothing
        # behind in the working directory.
        assert result.exit_code != 0
        assert list(Path.cwd().iterdir()) == []


def test_that_the_playground_still_accepts_an_explicit_directory(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PATH", "")

    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["widget", "playground", "--dir", "my-playground"])

        # Still rejected for the missing toolchain, but the flag is accepted.
        assert result.exit_code != 0
        assert "npm" in result.output


def test_that_a_port_check_sees_a_listener_bound_only_to_ipv6(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A dev server bound to [::1] leaves 127.0.0.1 bindable, so an IPv4-only
    # probe would report the port free and the playground would then collide.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")

    blocker = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        blocker.bind(("::1", 0))
    except OSError:
        pytest.skip("no IPv6 loopback on this host")
    blocker.listen(1)
    taken_port = blocker.getsockname()[1]

    try:
        with runner.isolated_filesystem():
            result = runner.invoke(
                cli, ["widget", "playground", "--port", str(taken_port), "--no-agent"]
            )

            assert result.exit_code != 0
            assert str(taken_port) in result.output
            assert "in use" in result.output.lower()
    finally:
        blocker.close()
