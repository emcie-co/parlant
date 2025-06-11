from abc import ABC, abstractmethod
import asyncio
import os
import random
from typing import Mapping, Tuple, Dict, List, Awaitable, Any, Sequence
from typing_extensions import override
import time
from together import AsyncTogether  # , Callable
import statistics
import openai

# Configuration
NUM_REPETITIONS = 10
BATCH_N = 3


async def retry_async(
    func_factory,
    max_retries: int = 3,
    delay: float = 1.0,
    retry_exceptions: tuple = (Exception,),
):
    """
    Retry an async function up to max_retries times if it raises an exception.
    func_factory: a zero-argument function that returns the coroutine to run.
    """
    for attempt in range(max_retries):
        try:
            return await func_factory()
        except retry_exceptions as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay)


async def with_warmup(coro_factory):
    # Perform two warm-up calls, discard results, then do the real call
    await coro_factory()
    await coro_factory()
    return await coro_factory()


async def gather_with_concurrency(batch_size: int, coros: Sequence[Awaitable[Any]]) -> List[Any]:
    """
    Run coroutines with at most batch_size concurrent tasks.
    Returns results in the same order as input.
    """
    semaphore = asyncio.Semaphore(batch_size)
    results = [None] * len(coros)

    async def sem_task(i, coro):
        async with semaphore:
            results[i] = await coro

    tasks = [asyncio.create_task(sem_task(i, coro)) for i, coro in enumerate(coros)]
    await asyncio.gather(*tasks)
    return results


def _randomize_prompt(prompt: str) -> str:
    random_prefix = str(random.randint(1, 1000000))
    return random_prefix + " " + prompt


class LLMRunner(ABC):
    @abstractmethod
    async def evaluate(self, prompt: str, max_tokens: int) -> Tuple[int, float]: ...


class TogetherLLMRunner(LLMRunner):
    def __init__(self, model_name):
        self._model_name = model_name
        self._together_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

    @override
    async def evaluate(self, prompt: str, max_tokens: int = 9999) -> Tuple[int, float]:
        start = time.perf_counter()
        response = await self._together_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._model_name,
            max_tokens=max_tokens,
        )
        time_elapsed = time.perf_counter() - start
        return response.usage.completion_tokens, time_elapsed  # type: ignore


class OpenaiLLMRunner(LLMRunner):
    def __init__(self, model_name):
        self._model_name = model_name
        self._openai_client = openai.AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @override
    async def evaluate(self, prompt: str, max_tokens: int = 9999) -> Tuple[int, float]:
        start = time.perf_counter()
        response = await self._openai_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._model_name,
            max_tokens=max_tokens,
        )
        time_elapsed = time.perf_counter() - start
        return response.usage.completion_tokens, time_elapsed  # type: ignore


async def run_benchmark():
    LLMRunners: Mapping[str, LLMRunner] = {
        "together-llama-3.1-8B": TogetherLLMRunner("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
        "together-llama-3.1-70B": TogetherLLMRunner("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
        "together-llama-3.1-405B": TogetherLLMRunner(
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        ),
        "together-llama-4-Maverick-17B": TogetherLLMRunner(
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
        ),
        "together-llama-4-Scout-17B": TogetherLLMRunner(
            "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        ),
        "together-llama-3.3-70B": TogetherLLMRunner("meta-llama/Llama-3.3-70B-Instruct-Turbo"),
        "openai-gpt-4o-2024-11-20": OpenaiLLMRunner("gpt-4o-2024-11-20"),
        "openai-gpt-4o-2024-08-06": OpenaiLLMRunner("gpt-4o-2024-08-06"),
        "openai-gpt-4o-mini": OpenaiLLMRunner("gpt-4o-mini"),
    }

    results = {}
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            base_prompt = f.read().strip()
    except FileNotFoundError:
        print("Error: prompt.txt not found. Please create this file with your base prompt.")
        return

    print(f"🚀 Starting LLM Token Generation Benchmark ({NUM_REPETITIONS} repetitions per test)...")
    print("=" * 50)

    for llm_name, llm_runner in LLMRunners.items():
        print(f"\nTesting {llm_name}...")
        try:
            # Test 1: Generate 1 token to measure time to first token
            print(f"  Measuring first token generation time ({NUM_REPETITIONS} times)...")
            first_token_times = []
            single_token_coros = [
                retry_async(
                    lambda llm_runner=llm_runner, base_prompt=base_prompt: with_warmup(
                        lambda: llm_runner.evaluate(
                            prompt=_randomize_prompt(base_prompt), max_tokens=1
                        )
                    )
                )
                for i in range(NUM_REPETITIONS)
            ]
            single_token_result = await gather_with_concurrency(BATCH_N, single_token_coros)
            for tokens_n, duration in single_token_result:
                assert tokens_n == 1
                first_token_times.append(duration)
            # Print individual first token times for each repetition
            for i, duration in enumerate(first_token_times):
                print(f"    First token repetition {i + 1}/{NUM_REPETITIONS}: {duration:.4f}s")
            avg_first_token_time = sum(first_token_times) / len(first_token_times)
            stdev_first_token_time = (
                statistics.stdev(first_token_times) if len(first_token_times) > 1 else 0.0
            )

            # Test 2: Generate many tokens to measure subsequent token speed
            print(f"  Measuring subsequent token generation speed ({NUM_REPETITIONS} times)...")
            # Run all repetitions concurrently using asyncio.gather
            coros = [
                retry_async(
                    lambda llm_runner=llm_runner, base_prompt=base_prompt: with_warmup(
                        lambda: llm_runner.evaluate(
                            prompt=_randomize_prompt(base_prompt), max_tokens=1024
                        )
                    )
                )
                for _ in range(NUM_REPETITIONS)
            ]
            full_generation_data = await gather_with_concurrency(BATCH_N, coros)
            tokens_many_list = [d[0] for d in full_generation_data]
            time_many_list = [d[1] for d in full_generation_data]
            for i, (tokens_many, time_many) in enumerate(full_generation_data):
                print(
                    f"    Repetition {i + 1}/{NUM_REPETITIONS}... {tokens_many} tokens in {time_many:.4f}s"
                )

            # Calculate average metrics from the multi-token generation test
            avg_total_tokens = sum(tokens_many_list) / len(tokens_many_list)
            avg_total_time = sum(time_many_list) / len(time_many_list)
            stdev_total_tokens = (
                statistics.stdev(tokens_many_list) if len(tokens_many_list) > 1 else 0.0
            )
            stdev_total_time = statistics.stdev(time_many_list) if len(time_many_list) > 1 else 0.0

            # Calculate the average time per subsequent token
            if avg_total_tokens > 1:
                avg_subsequent_token_time = (
                    avg_total_time - statistics.median(first_token_times)
                ) / (avg_total_tokens - 1)
            else:
                avg_subsequent_token_time = 0.0

            # Calculate stdev for subsequent token time
            # For each repetition, estimate subsequent token time as (time_many - first_token_time) / (tokens_many - 1)
            subsequent_token_times = []
            for i in range(len(tokens_many_list)):
                if tokens_many_list[i] > 1:
                    subsequent_token_times.append(
                        (time_many_list[i] - avg_first_token_time) / (tokens_many_list[i] - 1)
                    )
            avg_subsequent_token_time = (
                sum(subsequent_token_times) / len(subsequent_token_times)
                if subsequent_token_times
                else 0.0
            )
            stdev_subsequent_token_time = (
                statistics.stdev(subsequent_token_times) if len(subsequent_token_times) > 1 else 0.0
            )

            # Calculate tokens per second for each repetition (excluding first token)
            tokens_per_second_excl_first_list = [
                ((tokens_many_list[i] - 1) / (time_many_list[i] - first_token_times[i]))
                if tokens_many_list[i] > 1 and (time_many_list[i] - first_token_times[i]) > 0
                else 0.0
                for i in range(len(tokens_many_list))
            ]
            avg_tokens_per_second_excl_first = (
                sum(tokens_per_second_excl_first_list)
                / len([v for v in tokens_per_second_excl_first_list if v > 0])
                if any(v > 0 for v in tokens_per_second_excl_first_list)
                else 0.0
            )
            stdev_tokens_per_second_excl_first = (
                statistics.stdev([v for v in tokens_per_second_excl_first_list if v > 0])
                if len([v for v in tokens_per_second_excl_first_list if v > 0]) > 1
                else 0.0
            )

            # Store results
            results[llm_name] = {
                "first_token_time": avg_first_token_time,
                "median_first_token_time": avg_first_token_time,
                "first_token_time_stdev": stdev_first_token_time,
                "subsequent_token_time": avg_subsequent_token_time,
                "subsequent_token_time_stdev": stdev_subsequent_token_time,
                "total_tokens_generated": avg_total_tokens,
                "total_tokens_generated_stdev": stdev_total_tokens,
                "total_time": avg_total_time,
                "total_time_stdev": stdev_total_time,
                "tokens_per_second_excl_first": avg_tokens_per_second_excl_first,
                "tokens_per_second_excl_first_stdev": stdev_tokens_per_second_excl_first,
            }

            print(
                f"  ✅ Avg first token time: {avg_first_token_time:.4f}s (stdev: {stdev_first_token_time:.4f}s)"
            )
            print(
                f"  ✅ Avg subsequent token time: {avg_subsequent_token_time:.4f}s (stdev: {stdev_subsequent_token_time:.4f}s)"
            )
            print(
                f"  ✅ Avg total tokens: {avg_total_tokens:.1f} (stdev: {stdev_total_tokens:.2f})"
            )
            print(f"  ✅ Avg total time: {avg_total_time:.4f}s (stdev: {stdev_total_time:.4f}s)")

        except Exception as e:
            print(f"  ❌ Error testing {llm_name}: {e}")
            results[llm_name] = {"error": str(e)}

    # Print summary
    print("\n" + "=" * 50)
    print("📊 BENCHMARK RESULTS SUMMARY (AVERAGES)")
    print("=" * 50)

    successful_results = {k: v for k, v in results.items() if "error" not in v}

    if successful_results:
        # Sort by fastest first token time for clearer presentation
        print(
            f"{'LLM':<25} {'First Token (s)':<18} {'(stdev)':<10} "
            f"{'Subseq (s)':<14} {'(stdev)':<10} "
            f"{'Total Time (s)':<16} {'(stdev)':<10} "
            f"{'Tok/sec (excl)':<14} {'(stdev)':<10}"
        )
        print("-" * 140)
        for llm_name, metrics in successful_results.items():
            print(
                f"{llm_name:<25} {metrics['first_token_time']:<18.4f} {metrics['first_token_time_stdev']:<10.4f} "
                f"{metrics['subsequent_token_time']:<14.4f} {metrics['subsequent_token_time_stdev']:<10.4f} "
                f"{metrics['total_time']:<16.4f} {metrics['total_time_stdev']:<10.4f} "
                f"{metrics['tokens_per_second_excl_first']:<14.2f} {metrics['tokens_per_second_excl_first_stdev']:<10.2f}"
            )

        best_first_token = min(successful_results.items(), key=lambda x: x[1]["first_token_time"])
        best_subsequent = min(
            successful_results.items(), key=lambda x: x[1]["subsequent_token_time"]
        )

        print("\n🏆 Best Performance:")
        print(
            f"  ⏱️ Fastest first token: {best_first_token[0]} ({best_first_token[1]['first_token_time']:.4f}s)"
        )
        print(
            f"  ⚡ Fastest subsequent tokens: {best_subsequent[0]} ({best_subsequent[1]['subsequent_token_time']:.4f}s)"
        )

    error_results = {k: v for k, v in results.items() if "error" in v}
    if error_results:
        print("\n⚠️ Errors encountered:")
        for llm_name, error_info in error_results.items():
            print(f"  {llm_name}: {error_info['error']}")

    return results


def export_results_csv(results: Dict, filename: str = "llm_benchmark_results.csv"):
    import csv

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "llm_name",
            "avg_first_token_time",
            "stdev_first_token_time",
            "avg_subsequent_token_time",
            "stdev_subsequent_token_time",
            "avg_total_tokens",
            "stdev_total_tokens",
            "avg_total_time",
            "stdev_total_time",
            "avg_tokens_per_second_excl_first",
            "stdev_tokens_per_second_excl_first",
            "error",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for llm_name, metrics in results.items():
            row = {"llm_name": llm_name}
            if "error" in metrics:
                row["error"] = metrics["error"]
            else:
                row.update(
                    {
                        "avg_first_token_time": metrics["first_token_time"],
                        "stdev_first_token_time": metrics.get("first_token_time_stdev", ""),
                        "avg_subsequent_token_time": metrics["subsequent_token_time"],
                        "stdev_subsequent_token_time": metrics.get(
                            "subsequent_token_time_stdev", ""
                        ),
                        "avg_total_tokens": metrics["total_tokens_generated"],
                        "stdev_total_tokens": metrics.get("total_tokens_generated_stdev", ""),
                        "avg_total_time": metrics["total_time"],
                        "stdev_total_time": metrics.get("total_time_stdev", ""),
                        "avg_tokens_per_second_excl_first": metrics.get(
                            "tokens_per_second_excl_first", ""
                        ),
                        "stdev_tokens_per_second_excl_first": metrics.get(
                            "tokens_per_second_excl_first_stdev", ""
                        ),
                        "error": "",
                    }
                )
            writer.writerow(row)

    print(f"Results exported to {filename}")


if __name__ == "__main__":
    benchmark_results = asyncio.run(run_benchmark())
    # export_results_csv(benchmark_results)
