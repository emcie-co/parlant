from contextlib import asynccontextmanager
import copy
import json
from pathlib import Path
from pydoc import importfile
import tempfile
from typing import AsyncIterator

from pytest import fixture

from emcie.server.configuration_validator import ConfigurationFileValidator
from emcie.server.core.common import JSONSerializable


@asynccontextmanager
async def new_file_path() -> AsyncIterator[Path]:
    with tempfile.NamedTemporaryFile() as new_file:
        yield Path(new_file.name)


@fixture
async def valid_config() -> JSONSerializable:
    def create_valid_config() -> JSONSerializable:
        return {
            "agents": [{"name": "Default Agent"}],
            "guidelines": {
                "Default Agent": [
                    {
                        "when": "Ask to multiply two numbers",
                        "then": "use the multiply tool to provide the result",
                        "enabled_tools": ["multiply"],
                    }
                ]
            },
            "tools": {
                "multiply": {
                    "description": "Multiply two numbers",
                    "function_name": "multiply",
                    "module_path": importfile(str(tool_file)).__name__,
                    "parameters": {
                        "a": {"description": "first number", "type": "number"},
                        "b": {"description": "second number", "type": "number"},
                    },
                    "required": ["a", "b"],
                    "type": "python",
                }
            },
            "terminology":
            {
               "Default Agent": [
                    {
                        "name": "walnut",
                        "description": "a name of an altcoin",
                        "synonyms": ["Emcie's altcoin"],
                    }
                ]
            }
        }

    async with new_file_path() as tool_file:
        with open(tool_file, "w") as f:
            f.write("""def multiply(a, b): return a * b""")
        valid_config = create_valid_config()
    return valid_config


async def test_that_empty_config_is_valid() -> None:
    async with new_file_path() as config_file:
        with open(config_file, "w") as f:
            f.write(
                json.dumps(
                    {
                        "agents": [{"name": "Default Agent"}],
                        "guidelines": {"Default Agent": []},
                        "tools": {},
                        "terminology": {},
                    }
                )
            )
        assert ConfigurationFileValidator().validate(config_file) is True


async def test_that_a_valid_config_passes_validation(
    valid_config: JSONSerializable,
) -> None:
    async with new_file_path() as config_file:
        with open(config_file, "w") as f:
            f.write(json.dumps(valid_config))

        assert ConfigurationFileValidator().validate(config_file) is True


async def test_that_a_config_with_a_tool_whose_module_cannot_be_found_fails_validation(
    valid_config: JSONSerializable,
) -> None:
    async with new_file_path() as config_file, new_file_path() as tool_file:
        invalid_config: JSONSerializable = copy.deepcopy(valid_config)
        invalid_config["tools"]["multiply"][  # type: ignore
            "module_path"
        ] = "invalid.path.to.multiply"

        with open(config_file, "w") as f:
            f.write(json.dumps(invalid_config))

        assert ConfigurationFileValidator().validate(config_file) is False

        with open(tool_file, "w") as f:
            f.write("""def not_multiply(): return""")

        assert ConfigurationFileValidator().validate(config_file) is False


async def test_that_a_config_with_missing_mandatory_keys_in_guideline_fails_validation(
    valid_config: JSONSerializable,
) -> None:
    async with new_file_path() as config_file:
        invalid_config = copy.deepcopy(valid_config)
        del invalid_config["guidelines"]["Default Agent"][0]["when"]  # type: ignore

        with open(config_file, "w") as f:
            f.write(json.dumps(invalid_config))

        assert ConfigurationFileValidator().validate(config_file) is False

        invalid_config = copy.deepcopy(valid_config)
        del invalid_config["guidelines"]["Default Agent"][0]["then"]  # type: ignore

        with open(config_file, "w") as f:
            f.write(json.dumps(invalid_config))

        assert ConfigurationFileValidator().validate(config_file) is False


async def test_that_a_config_with_a_guideline_associated_to_a_nonexistent_tool_fails_validation(
    valid_config: JSONSerializable,
) -> None:
    async with new_file_path() as config_file:
        invalid_config = copy.deepcopy(valid_config)
        invalid_config["guidelines"]["Default Agent"][0]["enabled_tools"] = [  # type: ignore
            "nonexistent_tool"
        ]

        with open(config_file, "w") as f:
            f.write(json.dumps(invalid_config))

        assert ConfigurationFileValidator().validate(config_file) is False


async def test_that_guidelines_under_nonexistent_agent_fail_validation(
    valid_config: JSONSerializable,
) -> None:
    async with new_file_path() as config_file:
        invalid_config = copy.deepcopy(valid_config)
        invalid_config["guidelines"]["Nonexistent Agent"] = [  # type: ignore
            {"when": "Example condition", "then": "Example action"}
        ]

        with open(config_file, "w") as f:
            f.write(json.dumps(invalid_config))

        assert ConfigurationFileValidator().validate(config_file) is False


async def test_that_syntactically_invalid_json_fails_validation() -> None:
    async with new_file_path() as config_file:
        with open(config_file, "w") as f:
            f.write("{invalid_json: true,}")

        assert ConfigurationFileValidator().validate(config_file) is False


async def test_that_terms_under_nonexistent_agent_fail_validation(
    valid_config: JSONSerializable,
) -> None:
    async with new_file_path() as config_file:
        invalid_config = copy.deepcopy(valid_config)
        invalid_config["terminology"]["Nonexistent Agent"] = [  # type: ignore
            {"name": "Example term", "description": "Example description"}
        ]

        with open(config_file, "w") as f:
            f.write(json.dumps(invalid_config))

        assert ConfigurationFileValidator().validate(config_file) is False