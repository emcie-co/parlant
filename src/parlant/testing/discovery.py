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

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

from parlant.testing.suite import Suite


class DiscoveryError(Exception):
    """Error during test discovery."""

    pass


def discover_suites(paths: List[str]) -> List[Suite]:
    """Discover test suites from the given file/directory paths.

    Args:
        paths: List of file or directory paths to search.

    Returns:
        List of discovered Suite instances.

    Raises:
        DiscoveryError: If a test file fails to import.
    """
    suites: List[Suite] = []
    test_files = _collect_test_files(paths)

    for file_path in test_files:
        file_suites = _load_suites_from_file(file_path)
        suites.extend(file_suites)

    return suites


def _collect_test_files(paths: List[str]) -> List[Path]:
    """Collect all test_*.py files from the given paths."""
    test_files: List[Path] = []

    for path_str in paths:
        path = Path(path_str)

        if not path.exists():
            raise DiscoveryError(f"Path does not exist: {path}")

        if path.is_file():
            if path.name.startswith("test_") and path.suffix == ".py":
                test_files.append(path)
            else:
                raise DiscoveryError(f"File does not match test_*.py pattern: {path}")
        elif path.is_dir():
            # Recursively find all test_*.py files
            for py_file in path.rglob("test_*.py"):
                test_files.append(py_file)
        else:
            raise DiscoveryError(f"Invalid path: {path}")

    return sorted(set(test_files))  # Remove duplicates and sort


def _load_suites_from_file(file_path: Path) -> List[Suite]:
    """Load all Suite instances from a Python file.

    Args:
        file_path: Path to the Python file.

    Returns:
        List of Suite instances found in the file.

    Raises:
        DiscoveryError: If the file fails to import.
    """
    # Create a unique module name
    module_name = f"parlant_test_module_{file_path.stem}_{id(file_path)}"

    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise DiscoveryError(f"Could not load spec for: {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Find all Suite instances in the module
        suites: List[Suite] = []
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, Suite):
                suites.append(obj)

        return suites

    except Exception as e:
        # Clean up module if it was partially loaded
        if module_name in sys.modules:
            del sys.modules[module_name]

        raise DiscoveryError(
            f"Failed to import test file: {file_path}\nError: {type(e).__name__}: {e}"
        ) from e


def list_tests(suites: List[Suite]) -> List[str]:
    """List all test names from the given suites.

    Args:
        suites: List of Suite instances.

    Returns:
        List of test names.
    """
    tests: List[str] = []

    for suite in suites:
        for scenario in suite.get_scenarios():
            if scenario.repetitions > 1:
                for rep in range(1, scenario.repetitions + 1):
                    tests.append(f"{scenario.name}[rep_{rep}/{scenario.repetitions}]")
            else:
                tests.append(scenario.name)

    return tests
