import json
from typing import Any
from parlant.core.tools import Tool, ToolId, ToolParameterDescriptor, ToolParameterOptions


def format_type(descriptor_type: str) -> str:
    """Return the type-specific format suffix for the given descriptor type."""
    if descriptor_type == "datetime":
        return f"{descriptor_type}: year-month-day hour:minute:second"
    if descriptor_type == "date":
        return f"{descriptor_type}: year-month-day"
    if descriptor_type == "timedelta":
        return f"{descriptor_type}: hours:minutes:seconds"
    return descriptor_type


def get_param_spec(spec: tuple[ToolParameterDescriptor, ToolParameterOptions]) -> str:
    descriptor, options = spec

    result: dict[str, Any] = {"schema": {"type": format_type(descriptor["type"])}}

    if descriptor["type"] == "array":
        result["schema"]["items"] = {"type": format_type(descriptor["item_type"])}

        if enum := descriptor.get("enum"):
            result["schema"]["items"]["enum"] = enum
    else:
        if enum := descriptor.get("enum"):
            result["schema"]["enum"] = enum

    if options.description:
        result["description"] = options.description
    elif description := descriptor.get("description"):
        result["description"] = description

    if examples := descriptor.get("examples"):
        result["extraction_examples__only_for_reference"] = examples

    match options.source:
        case "any":
            result["acceptable_source"] = (
                "This argument can be extracted in the best way you think (context, tool results, customer input, etc.)"
            )
        case "context":
            result["acceptable_source"] = (
                "This argument can be extracted only from the context given in this prompt (tool results, interaction, variables, etc.)"
            )
        case "customer":
            result["acceptable_source"] = (
                "This argument must be provided by the customer in the interaction itself, and NEVER automatically guessed by you"
            )

    return json.dumps(result)


def get_tool_spec(t_id: ToolId, t: Tool) -> dict[str, Any]:
    return {
        "tool_name": t_id.to_string(),
        "description": t.description,
        "optional_arguments": {
            name: get_param_spec(spec)
            for name, spec in t.parameters.items()
            if name not in t.required
        },
        "required_parameters": {
            name: get_param_spec(spec) for name, spec in t.parameters.items() if name in t.required
        },
    }
