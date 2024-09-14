import asyncio
import os
import traceback
import httpx
from tests.e2e.test_server_cli import REASONABLE_AMOUNT_OF_TIME
from tests.e2e.test_utilities import (
    CLI_CLIENT_PATH,
    DEFAULT_AGENT_NAME,
    SERVER_ADDRESS,
    _TestContext,
    load_active_agent,
    run_server,
)

REASONABLE_AMOUNT_OF_TIME_FOR_TERM_CREATION = 0.25


async def test_that_a_term_can_be_created_with_synonyms(
    context: _TestContext,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> None:
    term_name = "guideline"
    description = "when and then statements"
    synonyms = "rule, principle"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = load_active_agent(home_dir=context.home_dir, agent_name=agent_name)

        exec_args = [
            "poetry",
            "run",
            "python",
            CLI_CLIENT_PATH.as_posix(),
            "--server",
            SERVER_ADDRESS,
            "terminology",
            "add",
            "--agent-id",
            agent["id"],
            "--name",
            term_name,
            "--description",
            description,
            "--synonyms",
            synonyms,
        ]

        result = await asyncio.create_subprocess_exec(*exec_args)
        await result.wait()

        assert result.returncode == os.EX_OK

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30),
        ) as client:
            try:
                terminology_response = await client.get(
                    f"{SERVER_ADDRESS}/terminology/{agent['id']}",
                )
                terminology_response.raise_for_status()

                assert len(terminology_response.json()["terms"]) == 1
                assert terminology_response.json()["terms"][0]["name"] == "guideline"

            except:
                traceback.print_exc()
                raise


async def test_that_a_term_can_be_created_without_synonyms(
    context: _TestContext,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> None:
    term_name = "guideline_no_synonyms"
    description = "simple guideline with no synonyms"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = load_active_agent(home_dir=context.home_dir, agent_name=agent_name)

        exec_args = [
            "poetry",
            "run",
            "python",
            CLI_CLIENT_PATH.as_posix(),
            "--server",
            SERVER_ADDRESS,
            "terminology",
            "add",
            "--agent-id",
            agent["id"],
            "--name",
            term_name,
            "--description",
            description,
        ]

        result = await asyncio.create_subprocess_exec(*exec_args)
        await result.wait()

        assert result.returncode == os.EX_OK

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30),
        ) as client:
            try:
                terminology_response = await client.get(
                    f"{SERVER_ADDRESS}/terminology/{agent['id']}/{term_name}",
                )
                terminology_response.raise_for_status()

                term = terminology_response.json()
                assert term["name"] == term_name
                assert term["description"] == description
                assert term["synonyms"] is None

            except:
                traceback.print_exc()
                raise


async def test_that_terms_can_be_listed(
    context: _TestContext,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> None:
    guideline_term_name = "guideline"
    tool_term_name = "tool"
    guideline_description = "when and then statements"
    tool_description = "techniuqe to fetch external data"
    guideline_synonyms = "rule, instruction"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = load_active_agent(home_dir=context.home_dir, agent_name=agent_name)

        exec_args_1 = [
            "poetry",
            "run",
            "python",
            CLI_CLIENT_PATH.as_posix(),
            "--server",
            SERVER_ADDRESS,
            "terminology",
            "add",
            "--agent-id",
            agent["id"],
            "--name",
            guideline_term_name,
            "--description",
            guideline_description,
            "--synonyms",
            guideline_synonyms,
        ]
        exec_args_2 = [
            "poetry",
            "run",
            "python",
            CLI_CLIENT_PATH.as_posix(),
            "--server",
            SERVER_ADDRESS,
            "terminology",
            "add",
            "--agent-id",
            agent["id"],
            "--name",
            tool_term_name,
            "--description",
            tool_description,
        ]

        await asyncio.create_subprocess_exec(*exec_args_1)
        result = await asyncio.create_subprocess_exec(*exec_args_2)
        await result.wait()

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30),
        ) as client:
            try:
                terminology_response = await client.get(
                    f"{SERVER_ADDRESS}/terminology/{agent['id']}",
                )
                terminology_response.raise_for_status()

                terms = terminology_response.json()["terms"]
                assert len(terms) == 2

                term_names = {term["name"] for term in terms}
                assert guideline_term_name in term_names
                assert tool_term_name in term_names

            except:
                traceback.print_exc()
                raise


async def test_that_a_term_can_be_deleted(
    context: _TestContext,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> None:
    name = "guideline_delete"
    description = "to be deleted"
    synonyms = "rule, principle"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = load_active_agent(home_dir=context.home_dir, agent_name=agent_name)

        exec_args = [
            "poetry",
            "run",
            "python",
            CLI_CLIENT_PATH.as_posix(),
            "--server",
            SERVER_ADDRESS,
            "terminology",
            "add",
            "--agent-id",
            agent["id"],
            "--name",
            name,
            "--description",
            description,
            "--synonyms",
            synonyms,
        ]
        result = await asyncio.create_subprocess_exec(*exec_args)
        await result.wait()

        assert result.returncode == os.EX_OK

        exec_args_delete = [
            "poetry",
            "run",
            "python",
            CLI_CLIENT_PATH.as_posix(),
            "--server",
            SERVER_ADDRESS,
            "terminology",
            "delete",
            "--agent-id",
            agent["id"],
            "--name",
            name,
        ]
        result_delete = await asyncio.create_subprocess_exec(*exec_args_delete)
        await result_delete.wait()

        assert result_delete.returncode == os.EX_OK

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30),
        ) as client:
            try:
                terminology_response = await client.get(
                    f"{SERVER_ADDRESS}/terminology/{agent['id']}",
                )
                terminology_response.raise_for_status()

                terms = terminology_response.json()["terms"]
                assert len(terms) == 0

            except:
                traceback.print_exc()
                raise


async def test_that_terms_are_loaded_on_server_startup(
    context: _TestContext,
    agent_name: str = DEFAULT_AGENT_NAME,
) -> None:
    term_name = "guideline_no_synonyms"
    description = "simple guideline with no synonyms"

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = load_active_agent(home_dir=context.home_dir, agent_name=agent_name)

        exec_args = [
            "poetry",
            "run",
            "python",
            CLI_CLIENT_PATH.as_posix(),
            "--server",
            SERVER_ADDRESS,
            "terminology",
            "add",
            "--agent-id",
            agent["id"],
            "--name",
            term_name,
            "--description",
            description,
        ]

        result_delete = await asyncio.create_subprocess_exec(*exec_args)
        await result_delete.wait()

        assert result_delete.returncode == os.EX_OK

    with run_server(context):
        await asyncio.sleep(REASONABLE_AMOUNT_OF_TIME)

        agent = load_active_agent(home_dir=context.home_dir, agent_name=agent_name)

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(30),
        ) as client:
            try:
                terminology_response = await client.get(
                    f"{SERVER_ADDRESS}/terminology/{agent['id']}/{term_name}",
                )
                terminology_response.raise_for_status()

                term = terminology_response.json()
                assert term["name"] == term_name
                assert term["description"] == description
                assert term["synonyms"] is None

            except:
                traceback.print_exc()
                raise
