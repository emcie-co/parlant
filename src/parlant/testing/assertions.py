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

from parlant.core.common import DefaultBaseModel
from parlant.core.nlp.service import NLPService


class NLPTestSchema(DefaultBaseModel):
    """Schema for NLP test responses."""

    reasoning: str
    answer: bool


NLP_TEST_PROMPT = """\
Given a context and a condition, determine whether the
condition applies with respect to the given context.
If the condition applies, the answer is true;
otherwise, the answer is false.

Context: ###
{context}
###

Condition: ###
{condition}
###

Output JSON structure: ###
{{
    "reasoning": <STRING>,
    "answer": <BOOL>
}}
###

Example #1: ###
{{
    "reasoning": "The condition holds because...",
    "answer": true
}}
###

Example #2: ###
{{
    "reasoning": "The condition doesn't hold because...",
    "answer": false
}}
###
"""


async def nlp_test(
    nlp_service: NLPService,
    context: str,
    condition: str,
) -> tuple[bool, str]:
    """Evaluate if a condition holds for the given context using NLP.

    Args:
        nlp_service: The NLP service to use for evaluation.
        context: The context to evaluate (e.g., agent response).
        condition: The condition to check.

    Returns:
        A tuple of (answer: bool, reasoning: str).
    """
    generator = await nlp_service.get_schematic_generator(NLPTestSchema)

    inference = await generator.generate(
        prompt=NLP_TEST_PROMPT.format(context=context, condition=condition),
        hints={"temperature": 0.0, "strict": True},
    )

    return inference.content.answer, inference.content.reasoning
