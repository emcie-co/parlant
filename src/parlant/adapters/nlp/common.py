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


from parlant.core.meter import Counter, Meter


def normalize_json_output(raw_output: str) -> str:
    json_start = raw_output.find("```json")

    if json_start != -1:
        json_start = json_start + 7
    else:
        json_start = 0

    json_end = raw_output[json_start:].rfind("```")

    if json_end == -1:
        json_end = len(raw_output[json_start:])

    return raw_output[json_start : json_start + json_end].strip()


_INPUT_TOKENS_COUNTER: Counter
_OUTPUT_TOKENS_COUNTER: Counter
_CACHED_TOKENS_COUNTER: Counter
_COUNTERS_INITIALIZED = False


async def record_llm_metrics(
    meter: Meter,
    model_name: str,
    schema_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> None:
    global _COUNTERS_INITIALIZED
    global _INPUT_TOKENS_COUNTER
    global _OUTPUT_TOKENS_COUNTER
    global _CACHED_TOKENS_COUNTER

    if not _COUNTERS_INITIALIZED:
        _INPUT_TOKENS_COUNTER = meter.create_counter(
            name="input_tokens",
            description="Number of input tokens sent to a LLM model",
        )
        _OUTPUT_TOKENS_COUNTER = meter.create_counter(
            name="output_tokens",
            description="Number of output tokens received from a LLM model",
        )
        _CACHED_TOKENS_COUNTER = meter.create_counter(
            name="cached_input_tokens",
            description="Number of input tokens served from cache for a LLM model",
        )

        _COUNTERS_INITIALIZED = True

    await _INPUT_TOKENS_COUNTER.increment(
        input_tokens,
        {"model_name": model_name, "schema_name": schema_name},
    )

    await _OUTPUT_TOKENS_COUNTER.increment(
        output_tokens,
        {"model_name": model_name, "schema_name": schema_name},
    )

    await _CACHED_TOKENS_COUNTER.increment(
        cached_input_tokens,
        {"model_name": model_name, "schema_name": schema_name},
    )
