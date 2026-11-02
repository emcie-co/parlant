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

"""Static guardrails for the internal metadata ownership boundary."""

import ast
from pathlib import Path
from typing import Iterable


_REPOSITORY_ROOT = Path(__file__).parents[3]
_SOURCE_ROOT = _REPOSITORY_ROOT / "src" / "parlant"
_COMPASS_ROOTS = (_SOURCE_ROOT / "core" / "engines" / "compass",)


def _python_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        yield from root.rglob("*.py")


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _literal_string(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _contains_metadata_argument(node: ast.AST) -> bool:
    for descendant in ast.walk(node):
        if isinstance(descendant, ast.Dict):
            if any(_literal_string(key) == "metadata" for key in descendant.keys):
                return True
        elif isinstance(descendant, ast.keyword) and descendant.arg == "metadata":
            return True
    return False


def test_compass_does_not_replace_complete_session_metadata() -> None:
    violations: list[str] = []

    for path in _python_files(_COMPASS_ROOTS):
        for node in ast.walk(_tree(path)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update_session"
            ):
                continue

            update_arguments: list[ast.AST] = [*node.args[1:], *node.keywords]
            if any(_contains_metadata_argument(argument) for argument in update_arguments):
                violations.append(f"{path.relative_to(_REPOSITORY_ROOT)}:{node.lineno}")

    assert violations == [], (
        "Compass must stage internal session metadata or use the CAS-safe internal "
        f"patch capability; whole-map update_session calls found at {violations}"
    )


def test_production_does_not_call_public_metadata_methods_for_the_reserved_root() -> None:
    violations: list[str] = []

    for path in _python_files((_SOURCE_ROOT,)):
        for node in ast.walk(_tree(path)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"set_metadata", "unset_metadata"}
            ):
                continue

            key_argument: ast.expr | None = None
            if len(node.args) >= 2:
                key_argument = node.args[1]
            else:
                key_argument = next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg in {"key", "metadata_key"}
                    ),
                    None,
                )

            if _literal_string(key_argument) == "__parlant__":
                violations.append(f"{path.relative_to(_REPOSITORY_ROOT)}:{node.lineno}")

    assert violations == [], (
        "Internal code must use a trusted internal metadata operation; public "
        f"reserved-root mutations found at {violations}"
    )
