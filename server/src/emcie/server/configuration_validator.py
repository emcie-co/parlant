import importlib
import json
from pathlib import Path
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
                                "required": {"type": "array", "items": {"type": "string"}},
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
                }},
            "required": ["agents", "guidelines", "tools", "terminology"],
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

    def validate_tools(
        self,
        config: JSONSerializable,
    ) -> None:
        config = typing.cast(dict[str, typing.Any], config)

        for tool_name, tool_info in config["tools"].items():
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

        tools = set(config["tools"].keys())
        agents = set(agent["name"] for agent in config["agents"])
        for agent, guidelines in config["guidelines"].items():
            if agent not in agents:
                raise ValidationError(f'Agent "{agent}" does not exist.')

            for guideline in guidelines:
                if "enabled_tools" in guideline:
                    for tool_name in guideline["enabled_tools"]:
                        if tool_name not in tools:
                            raise ValidationError(
                                f'Tool "{tool_name}" listed in "enabled_tools"'
                                ' does not exist in "tools".'
                            )

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
            self.validate_tools(config)
            self.validate_guidelines(config)
            self.validate_terminology(config)
            return True
        except Exception as e:
            logger.error(f"Configuration file invalid: {str(e)}")
            return False
