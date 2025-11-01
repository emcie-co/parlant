import json
import os
from pathlib import Path
from typing import Any, Mapping, cast
import aiofiles
from typing_extensions import override

from parlant.core.async_utils import safe_gather
from parlant.core.common import generate_id
from parlant.core.tracer import Tracer
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.nlp.generation import T, SchematicGenerationResult, SchematicGenerator
from parlant.core.nlp.tokenization import EstimatingTokenizer


class DataCollectingSchematicGenerator(SchematicGenerator[T]):
    """A schematic generator that collects data during generation."""

    def __init__(
        self,
        wrapped_generator: SchematicGenerator[T],
        tracer: Tracer,
    ) -> None:
        self._wrapped_generator = wrapped_generator
        self._tracer = tracer

        if path := os.environ.get("PARLANT_DATA_COLLECTION_PATH"):
            self._base_path = Path(path)
        else:
            self._base_path = Path("./data-collection")

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        result = await self._wrapped_generator.generate(prompt=prompt, hints=hints)

        path = self._base_path

        if scope := self._tracer.get_attribute("scope"):
            path = path / cast(str, scope)

        if self._tracer.get_attribute("session_id"):
            session_id = self._tracer.get_attribute("session_id")
            path = path / f"Session_{session_id}"

        if request_id := self._tracer.get_attribute("request_id"):
            path = path / f"R{request_id}"

        if iteration := self._tracer.get_attribute("engine_iteration"):
            path = path / f"Iteration_{iteration}"

        path.mkdir(parents=True, exist_ok=True)

        generation_id = generate_id()

        prompt_path = path / f"{self._wrapped_generator.schema.__name__}_{generation_id}.prompt.txt"
        completion_path = (
            path / f"{self._wrapped_generator.schema.__name__}_{generation_id}.completion.txt"
        )
        usage_path = path / f"{self._wrapped_generator.schema.__name__}_{generation_id}.usage.txt"

        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        async with (
            aiofiles.open(prompt_path, "w", encoding="utf-8") as prompt_file,
            aiofiles.open(completion_path, "w", encoding="utf-8") as completion_file,
            aiofiles.open(usage_path, "w", encoding="utf-8") as usage_file,
        ):
            usage_info = json.dumps(
                {
                    "model": result.info.model,
                    "duration": result.info.duration,
                    "input_tokens": result.info.usage.input_tokens,
                    "cached_input_tokens": result.info.usage.extra
                    and result.info.usage.extra.get("cached_input_tokens", 0)
                    or 0,
                    "output_tokens": result.info.usage.output_tokens,
                },
                indent=2,
            )

            await safe_gather(
                prompt_file.write(prompt),
                completion_file.write(result.content.model_dump_json(indent=2)),
                usage_file.write(usage_info),
            )

        return result

    @property
    @override
    def id(self) -> str:
        return self._wrapped_generator.id

    @property
    @override
    def max_tokens(self) -> int:
        return self._wrapped_generator.max_tokens

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._wrapped_generator.tokenizer
