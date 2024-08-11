import importlib
import json
from pathlib import Path
import traceback
import typing
from jsonschema import ValidationError, validate
from loguru import logger

from emcie.server.core.common import JSONSerializable


class ConfigurationFileValidator:
    def __init__(
        self,
    ) -> None:
        self.schema = {
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                },
                "guidelines": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "when": {"type": "string"},
                                "then": {"type": "string"},
                                "enabled_tools": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["when", "then"],
                        },
                    },
                },
                "services": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["plugin"]},
                                    "name": {"type": "string"},
                                    "url": {"type": "string"},
                                },
                                "required": ["type", "name", "url"],
                            },
                            {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["local"]},
                                    "tools": {
                                        "type": "object",
                                        "patternProperties": {
                                            "^[a-zA-Z0-9_]+$": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "function_name": {"type": "string"},
                                                    "module_path": {"type": "string"},
                                                    "parameters": {
                                                        "type": "object",
                                                        "additionalProperties": {
                                                            "type": "object",
                                                            "properties": {
                                                                "description": {"type": "string"},
                                                                "type": {
                                                                    "enum": [
                                                                        "number",
                                                                        "string",
                                                                        "integer",
                                                                        "boolean",
                                                                        "array",
                                                                        "object",
                                                                    ]
                                                                },
                                                            },
                                                            "required": ["description", "type"],
                                                        },
                                                    },
                                                    "required": {
                                                        "type": "array",
                                                        "items": {"type": "string"},
                                                    },
                                                },
                                                "required": [
                                                    "description",
                                                    "function_name",
                                                    "module_path",
                                                    "parameters",
                                                    "type",
                                                ],
                                            }
                                        },
                                    },
                                },
                                "required": ["type", "tools"],
                            },
                        ]
                    },
                },
                "terminology": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "synonyms": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["name", "description"],
                        },
                    },
                },
            },
            "required": ["agents", "guidelines", "services"],
        }

    def validate_and_load_config_json(
        self,
        config_file: Path,
    ) -> JSONSerializable:
        try:
            config: JSONSerializable = json.loads(config_file.read_text())
            return config
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e.msg} at line {e.lineno}, column {e.colno}")

    def validate_json_schema(
        self,
        config: JSONSerializable,
    ) -> None:
        validate(instance=config, schema=self.schema)

    def validate_local_tools(
        self,
        config: JSONSerializable,
    ) -> None:
        config = typing.cast(dict[str, typing.Any], config)

        for service in config["services"]:
            if service["type"] != "local":
                continue

            for tool_name, tool_info in service["tools"].items():
                module_path = tool_info["module_path"]
                try:
                    module = importlib.import_module(module_path)
                except ModuleNotFoundError:
                    raise ValidationError(
                        f'Module path "{module_path}" for tool "{tool_name}" does not exist.'
                    )
                function_name = tool_info["function_name"]
                try:
                    getattr(module, function_name)
                except Exception:
                    raise ValidationError(
                        f'The function "{function_name}" for tool "{tool_name}" '
                        f'does not exist in the "{module_path}" module.'
                    )

    def validate_guidelines(
        self,
        config: JSONSerializable,
    ) -> None:
        config = typing.cast(dict[str, typing.Any], config)

        agents = set(agent["name"] for agent in config["agents"])
        for agent, guidelines in config["guidelines"].items():
            if agent not in agents:
                raise ValidationError(f'Agent "{agent}" does not exist.')

    def validate_terminology(
        self,
        config: JSONSerializable,
    ) -> None:
        config = typing.cast(dict[str, typing.Any], config)

        agents = set(agent["name"] for agent in config["agents"])
        for agent in config["terminology"]:
            if agent not in agents:
                raise ValidationError(f'Agent "{agent}" does not exist.')

    def validate(self, config_file: Path) -> bool:
        try:
            config = self.validate_and_load_config_json(config_file)
            self.validate_json_schema(config)
            self.validate_local_tools(config)
            self.validate_guidelines(config)
            self.validate_terminology(config)
            return True
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Configuration file invalid: {e.__class__.__name__}({str(e)})")
            return False
