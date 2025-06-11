from abc import ABC, abstractmethod
import asyncio
import os
import random
from typing import Mapping, Tuple, Dict
from typing_extensions import override
import time
from together import AsyncTogether  # , Callable

# Configuration
NUM_REPETITIONS = 1


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


import statistics


async def run_benchmark():
    LLMRunners: Mapping[str, LLMRunner] = {
        "together-llama-3.1-8B": TogetherLLMRunner("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"),
        "together-llama-3.1-70B": TogetherLLMRunner("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
        "together-llama-3.1-405B": TogetherLLMRunner(
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
        ),
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
            single_token_result = await asyncio.gather(
                *[
                    llm_runner.evaluate(prompt=_randomize_prompt(base_prompt), max_tokens=1)
                    for i in range(NUM_REPETITIONS)
                ]
            )
            for tokens_n, duration in single_token_result:
                assert tokens_n == 1
                first_token_times.append(duration)
            avg_first_token_time = sum(first_token_times) / len(first_token_times)
            stdev_first_token_time = (
                statistics.stdev(first_token_times) if len(first_token_times) > 1 else 0.0
            )

            # Test 2: Generate many tokens to measure subsequent token speed
            print(f"  Measuring subsequent token generation speed ({NUM_REPETITIONS} times)...")
            # Run all repetitions concurrently using asyncio.gather
            coros = [
                llm_runner.evaluate(prompt=_randomize_prompt(base_prompt), max_tokens=1024)
                for _ in range(NUM_REPETITIONS)
            ]
            full_generation_data = await asyncio.gather(*coros)
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
                avg_subsequent_token_time = (avg_total_time - avg_first_token_time) / (
                    avg_total_tokens - 1
                )
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

            # Calculate seconds per 100 tokens for each repetition
            seconds_per_100_tokens_list = [
                (time_many_list[i] / tokens_many_list[i]) * 100 if tokens_many_list[i] > 0 else 0.0
                for i in range(len(tokens_many_list))
            ]
            avg_seconds_per_100_tokens = sum(seconds_per_100_tokens_list) / len(
                seconds_per_100_tokens_list
            )
            stdev_seconds_per_100_tokens = (
                statistics.stdev(seconds_per_100_tokens_list)
                if len(seconds_per_100_tokens_list) > 1
                else 0.0
            )

            # Store results
            results[llm_name] = {
                "first_token_time": avg_first_token_time,
                "first_token_time_stdev": stdev_first_token_time,
                "subsequent_token_time": avg_subsequent_token_time,
                "subsequent_token_time_stdev": stdev_subsequent_token_time,
                "total_tokens_generated": avg_total_tokens,
                "total_tokens_generated_stdev": stdev_total_tokens,
                "total_time": avg_total_time,
                "total_time_stdev": stdev_total_time,
                "seconds_per_100_tokens": avg_seconds_per_100_tokens,
                "seconds_per_100_tokens_stdev": stdev_seconds_per_100_tokens,
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
            print(
                f"  ✅ Avg seconds per 100 tokens: {avg_seconds_per_100_tokens:.2f} (stdev: {stdev_seconds_per_100_tokens:.2f})"
            )

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
            f"{'LLM':<25} {'First Token (s)':<18} {'(stdev)':<10} {'Subseq (s)':<14} {'(stdev)':<10} {'Sec/100t':<10} {'(stdev)':<10}"
        )
        print("-" * 100)
        for llm_name, metrics in successful_results:
            print(
                f"{llm_name:<25} {metrics['first_token_time']:<18.4f} {metrics['first_token_time_stdev']:<10.4f} "
                f"{metrics['subsequent_token_time']:<14.4f} {metrics['subsequent_token_time_stdev']:<10.4f} "
                f"{metrics['seconds_per_100_tokens']:<10.2f} {metrics['seconds_per_100_tokens_stdev']:<10.2f}"
            )

        best_first_token = min(successful_results.items(), key=lambda x: x[1]["first_token_time"])
        best_subsequent = min(
            successful_results.items(), key=lambda x: x[1]["subsequent_token_time"]
        )
        best_efficiency = min(
            successful_results.items(), key=lambda x: x[1]["seconds_per_100_tokens"]
        )

        print("\n🏆 Best Performance:")
        print(
            f"  ⏱️ Fastest first token: {best_first_token[0]} ({best_first_token[1]['first_token_time']:.4f}s)"
        )
        print(
            f"  ⚡ Fastest subsequent tokens: {best_subsequent[0]} ({best_subsequent[1]['subsequent_token_time']:.4f}s)"
        )
        print(
            f"   best efficiency: {best_efficiency[0]} ({best_efficiency[1]['seconds_per_100_tokens']:.2f} sec/100 tokens)"
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
            "avg_seconds_per_100_tokens",
            "stdev_seconds_per_100_tokens",
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
                        "avg_seconds_per_100_tokens": metrics["seconds_per_100_tokens"],
                        "stdev_seconds_per_100_tokens": metrics.get(
                            "seconds_per_100_tokens_stdev", ""
                        ),
                        "error": "",
                    }
                )
            writer.writerow(row)

    print(f"Results exported to {filename}")


if __name__ == "__main__":
    benchmark_results = asyncio.run(run_benchmark())
    # export_results_csv(benchmark_results)
