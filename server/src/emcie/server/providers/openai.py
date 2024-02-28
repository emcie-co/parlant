from typing import Any, AsyncIterator, Iterable, List
from openai import AsyncOpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
import os
import json

from emcie.server.models import TextGenerationModel
from emcie.server.threads import Message


class GPT(TextGenerationModel):
    def __init__(
        self,
        model_id: str,
    ) -> None:
        self.client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model_id = model_id

    async def generate_text(
        self,
        messages: Iterable[Message],
        skills: Iterable[Any],
    ) -> AsyncIterator[str]:
        converted_messages = self._convert_messages(messages)

        specs = [s["spec"] for s in skills]

        if skills:
            response = await self.client.chat.completions.create(
                messages=converted_messages,
                model=self.model_id,
                stream=True,
                tools=specs,
                tool_choice="auto",
            )
        else:
            response = await self.client.chat.completions.create(
                messages=converted_messages,
                model=self.model_id,
                stream=True,
            )

        tool_call_id = ""
        function_name = ""
        function_args = ""

        async for x in response:
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
            function_args_parsed = json.loads(function_args)
            skill = [s for s in skills if s["spec"]["function"]["name"] == function_name][0]
            ret_val = skill["func"](**function_args_parsed)

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
                stream=True,
            )

            async for x in response:
                yield x.choices[0].delta.content or ""

    def _convert_messages(self, messages: Iterable[Message]) -> List[ChatCompletionMessageParam]:
        return [{"role": m.role, "content": m.content} for m in messages]  # type: ignore
