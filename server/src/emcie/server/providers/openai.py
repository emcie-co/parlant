# Copyright (c) 2024 Emcie
# All rights reserved.
#
# This file and its contents are the property of Emcie and are strictly confidential.
# No part of this file may be reproduced, distributed, or transmitted in any form or by any means,
# including photocopying, recording, or other electronic or mechanical methods,
# without the prior written permission of Emcie.
#
# Website: https://emcie.co
from typing import Any, AsyncIterator, Iterable, List
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import os
import json
from loguru import logger

from emcie.server.models import TextGenerationModel
from emcie.server.threads import Message


class GPT(TextGenerationModel):
    def __init__(
        self,
        model_id: str,
    ) -> None:
        self.model_id = model_id
        self.client = self.setup_client()

    def setup_client(self) -> AsyncOpenAI:
        raise NotImplementedError()

    def _create_system_message(self, rules: Iterable[Any]) -> str:
        prompt = "Instructions:\n"

        num = 1

        for rule in rules:
            prompt += f"RULE {num}:###\nWhen: {rule['when']}.\nThen: {rule['then']}\n###\n"
            num += 1

        return prompt

    async def generate_text(
        self,
        messages: Iterable[Message],
        skills: Iterable[Any],
        rules: Iterable[Any],
    ) -> AsyncIterator[str]:
        temperature = 0.3

        converted_messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": self._create_system_message(rules)}
        ] + self._convert_messages(
            messages
        )  # type: ignore

        specs = [s["spec"] for s in skills]

        if skills:
            response = await self.client.chat.completions.create(
                messages=converted_messages,
                model=self.model_id,
                temperature=temperature,
                stream=True,
                tools=specs,
                tool_choice="auto",
            )
        else:
            response = await self.client.chat.completions.create(
                messages=converted_messages,
                model=self.model_id,
                temperature=temperature,
                stream=True,
            )

        tool_call_id = ""
        function_name = ""
        function_args = ""

        async for x in response:
            if not x.choices:
                continue
            # TODO test parallel function calls
            if x.choices[0].delta.tool_calls:
                tool_call = x.choices[0].delta.tool_calls[0]
                if tool_call.id:
                    tool_call_id = tool_call.id
                if not tool_call.function:
                    continue
                if tool_call.function.name:
                    function_name += tool_call.function.name
                elif tool_call.function.arguments:
                    function_args += tool_call.function.arguments
            else:
                yield x.choices[0].delta.content or ""

        if function_name:
            logger.info(f"Running function {function_name} with args {function_args}")
            function_args_parsed = json.loads(function_args)
            skill = [s for s in skills if s["spec"]["function"]["name"] == function_name][0]

            try:
                ret_val = skill["func"](**function_args_parsed)
                logger.info(f"{function_name} returned: {ret_val}")
            except Exception as exc:
                logger.error(f"Error while running tool {function_name}: {exc}")
                ret_val = "Sorry, I encountered an error"

            converted_messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "function": {
                                "name": function_name,
                                "arguments": function_args,
                            },
                            "type": "function",
                        }
                    ],
                }
            )

            converted_messages.append(
                {
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "content": json.dumps({"result": ret_val}),
                }
            )

            response = await self.client.chat.completions.create(
                messages=converted_messages,
                model=self.model_id,
                temperature=temperature,
                stream=True,
            )

            async for x in response:
                if not x.choices:
                    continue
                yield x.choices[0].delta.content or ""

    def _convert_messages(self, messages: Iterable[Message]) -> List[ChatCompletionMessageParam]:
        return [{"role": m.role, "content": m.content} for m in messages]  # type: ignore


class OpenAIGPT(GPT):
    def __init__(
        self,
        model_id: str,
    ) -> None:
        super().__init__(model_id=model_id)

    def setup_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


class AzureGPT(GPT):
    def __init__(
        self,
        model_id: str,
    ) -> None:
        super().__init__(model_id=model_id)

    def setup_client(self) -> AsyncOpenAI:
        return AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            base_url=os.environ["AZURE_OPENAI_URL"] + f"/deployments/{self.model_id}",
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        )
