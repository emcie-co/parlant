from typing import Any, Sequence

from lagom import Container
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.tool_running_action_detector import ToolRunningActionDetector
from parlant.core.tools import LocalToolService, ToolId


async def base_test_tool_running_action_detector(
    container: Container,
    guideline: GuidelineContent,
    tools: Sequence[dict[str, Any]],
    is_tool_running: bool,
) -> None:
    local_tool_service = container[LocalToolService]
    tool_action_detector = container[ToolRunningActionDetector]

    local_tools = [await local_tool_service.create_tool(**tool) for tool in tools]

    result = await tool_action_detector.detect_if_tool_running(
        guideline=guideline,
        tool_ids=[ToolId(service_name="local", tool_name=tool.name) for tool in local_tools],
    )
    assert result.is_tool_running_only == is_tool_running


async def test_that_guideline_with_action_that_only_run_tool_is_detected(
    container: Container,
) -> None:
    tool: dict[str, Any] = {
        "name": "get_available_toppings",
        "description": "get all available toppings",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    }
    tools = [tool]
    guideline = GuidelineContent(
        condition="The customer asks about vegetarian options",
        action="get all vegetarian pizza toppings options",
    )
    is_tool_running = True

    await base_test_tool_running_action_detector(
        container,
        guideline,
        tools,
        is_tool_running,
    )


async def test_that_guideline_with_action_that_only_run_several_tools_is_detected(
    container: Container,
) -> None:
    deactivate_account_tool: dict[str, Any] = {
        "name": "deactivate_account",
        "description": "Disables the user's account and prevents future logins.",
        "module_path": "tools.user_management",
        "parameters": {
            "user_id": {
                "type": "string",
                "description": "The unique identifier of the user.",
            },
        },
        "required": ["user_id"],
    }

    revoke_sessions_tool: dict[str, Any] = {
        "name": "revoke_sessions",
        "description": "Terminates all currently active sessions for the user.",
        "module_path": "tools.session_control",
        "parameters": {
            "user_id": {
                "type": "string",
                "description": "The unique identifier of the user.",
            },
        },
        "required": ["user_id"],
    }

    tools = [deactivate_account_tool, revoke_sessions_tool]

    guideline = GuidelineContent(
        condition="Suspicious activity detected",
        action="Deactivate the user's account and revoke all active sessions",
    )

    is_tool_running = True

    await base_test_tool_running_action_detector(
        container,
        guideline,
        tools,
        is_tool_running,
    )


async def test_that_guideline_with_action_that_not_only_require_running_tools_is_not_detected(
    container: Container,
) -> None:
    deactivate_account_tool: dict[str, Any] = {
        "name": "deactivate_account",
        "description": "Disables the user's account and prevents future logins.",
        "module_path": "tools.user_management",
        "parameters": {
            "user_id": {
                "type": "string",
                "description": "The unique identifier of the user.",
            },
        },
        "required": ["user_id"],
    }

    revoke_sessions_tool: dict[str, Any] = {
        "name": "revoke_sessions",
        "description": "Terminates all currently active sessions for the user.",
        "module_path": "tools.session_control",
        "parameters": {
            "user_id": {
                "type": "string",
                "description": "The unique identifier of the user.",
            },
        },
        "required": ["user_id"],
    }

    tools = [deactivate_account_tool, revoke_sessions_tool]

    guideline = GuidelineContent(
        condition="Suspicious activity detected",
        action="Deactivate the user's account and revoke all active sessions and reflect the situation to the user",
    )
    is_tool_running = False

    await base_test_tool_running_action_detector(
        container,
        guideline,
        tools,
        is_tool_running,
    )


async def test_that_guideline_with_action_that_require_a_tool_but_unrelated_associated_tool_is_not_detected(
    container: Container,
) -> None:
    tool: dict[str, Any] = {
        "name": "check_customer_info",
        "description": "Retrieves stored information about the customer, such as name, contact details, and account status.",
        "module_path": "tools.customer_data",
        "parameters": {
            "customer_id": {
                "type": "string",
                "description": "The unique identifier of the customer whose information should be retrieved.",
            },
        },
        "required": ["customer_id"],
    }
    tools = [tool]
    guideline = GuidelineContent(
        condition="need to verify the customer's identity",
        action="send a verification code",
    )
    is_tool_running = False

    await base_test_tool_running_action_detector(
        container,
        guideline,
        tools,
        is_tool_running,
    )


async def test_that_guideline_with_action_that_require_running_tools_and_telling_the_user_something_is_not_detected(
    container: Container,
) -> None:
    tool: dict[str, Any] = {
        "name": "get_available_toppings",
        "description": "get all available toppings",
        "module_path": "tests.tool_utilities",
        "parameters": {},
        "required": [],
    }
    tools = [tool]
    guideline = GuidelineContent(
        condition="The customer asks about vegetarian options",
        action="list all vegetarian pizza toppings options",
    )
    is_tool_running = False

    await base_test_tool_running_action_detector(
        container,
        guideline,
        tools,
        is_tool_running,
    )
