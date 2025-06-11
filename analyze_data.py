import json
import pandas as pd
from collections import defaultdict


def analyze_model_pass_rates(file_path):
    """
    Analyze test results by model and calculate pass rates and average durations.
    Each unique test is counted once per model, with averaged results.
    """
    # Structure: model_name -> test_name -> list of results
    model_test_results = defaultdict(lambda: defaultdict(list))

    with open(file_path, "r") as file:
        for line in file:
            stripped_line = line.strip()
            if stripped_line:
                try:
                    test_record = json.loads(stripped_line)
                    model_name = test_record.get("model_name", "unknown")
                    test_name = test_record.get("test_name", "unknown")
                    result = test_record.get("result", "unknown")
                    duration = test_record.get("duration", -1.0)

                    # Store each test run result
                    model_test_results[model_name][test_name].append(
                        {"result": result, "duration": duration}
                    )

                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue

    # Now aggregate results per test for each model
    model_stats = {}

    for model_name, test_results in model_test_results.items():
        total_tests = 0
        total_pass_rate = 0.0
        all_durations = []

        for test_name, runs in test_results.items():
            total_tests += 1

            # Calculate pass rate for this test (fraction of runs that passed)
            passed_runs = sum(1 for run in runs if run["result"] == "passed")
            test_pass_rate = passed_runs / len(runs) if runs else 0.0
            total_pass_rate += test_pass_rate

            # Collect positive durations from all runs of this test
            positive_durations = [run["duration"] for run in runs if run["duration"] > 0]
            if positive_durations:
                # Average duration for this test
                avg_test_duration = sum(positive_durations) / len(positive_durations)
                all_durations.append(avg_test_duration)

        model_stats[model_name] = {
            "total_tests": total_tests,
            "avg_pass_rate": (total_pass_rate / total_tests * 100) if total_tests > 0 else 0.0,
            "durations": all_durations,
        }

    return model_stats


def create_pass_rate_table(model_stats):
    """
    Create a formatted table showing pass rates and average durations by model.
    """
    # Prepare data for DataFrame
    table_data = []

    for model_name, stats in model_stats.items():
        total_tests = stats["total_tests"]
        avg_pass_rate = stats["avg_pass_rate"]
        durations = stats["durations"]

        # Calculate average duration across all tests for this model
        if durations:
            avg_duration = sum(durations) / len(durations)
        else:
            avg_duration = None  # No positive durations found

        table_data.append(
            {
                "Model Name": model_name,
                "Unique Tests": total_tests,
                "Avg Pass Rate (%)": round(avg_pass_rate, 1),
                "Avg Duration (s)": round(avg_duration, 3) if avg_duration is not None else "N/A",
            }
        )

    # Create DataFrame and sort by pass rate (descending)
    df = pd.DataFrame(table_data)
    df = df.sort_values("Avg Pass Rate (%)", ascending=False)

    return df


def print_formatted_table(df):
    """
    Print a nicely formatted table.
    """
    print("\n" + "=" * 80)
    print("MODEL PASS RATE AND PERFORMANCE ANALYSIS (Averaged per Test)")
    print("=" * 80)

    # Print the table with nice formatting
    print(
        df.to_string(
            index=False,
            formatters={
                "Avg Pass Rate (%)": lambda x: f"{x:>6.1f}%",
                "Avg Duration (s)": lambda x: f"{x:>8}" if x == "N/A" else f"{x:>8.3f}",
            },
        )
    )

    print("=" * 80)

    # Print summary statistics
    total_models = len(df)
    avg_pass_rate = df["Avg Pass Rate (%)"].mean()
    best_model = df.iloc[0] if not df.empty else None
    worst_model = df.iloc[-1] if not df.empty else None

    # Calculate average duration across all models (excluding N/A values)
    numeric_durations = [x for x in df["Avg Duration (s)"] if x != "N/A"]
    if numeric_durations:
        overall_avg_duration = sum(numeric_durations) / len(numeric_durations)
    else:
        overall_avg_duration = None

    print("\nSUMMARY:")
    print(f"Total Models: {total_models}")
    print(f"Average Pass Rate: {avg_pass_rate:.1f}%")

    if overall_avg_duration is not None:
        print(f"Overall Average Duration: {overall_avg_duration:.3f}s")

    if best_model is not None:
        print(
            f"Best Performing Model: {best_model['Model Name']} ({best_model['Avg Pass Rate (%)']}%)"
        )

    if worst_model is not None and total_models > 1:
        print(
            f"Worst Performing Model: {worst_model['Model Name']} ({worst_model['Avg Pass Rate (%)']}%)"
        )

    # Find fastest and slowest models
    if numeric_durations:
        fastest_model = df[df["Avg Duration (s)"] != "N/A"].loc[df["Avg Duration (s)"].idxmin()]
        slowest_model = df[df["Avg Duration (s)"] != "N/A"].loc[df["Avg Duration (s)"].idxmax()]

        if len(df[df["Avg Duration (s)"] != "N/A"]) > 1:
            print(
                f"Fastest Model: {fastest_model['Model Name']} ({fastest_model['Avg Duration (s)']}s)"
            )
            print(
                f"Slowest Model: {slowest_model['Model Name']} ({slowest_model['Avg Duration (s)']}s)"
            )


# Main execution
if __name__ == "__main__":
    # Replace 'paste.txt' with your actual file path
    file_path = "parlant_test_results.jsonl"

    try:
        model_stats = analyze_model_pass_rates(file_path)

        if model_stats:
            df = create_pass_rate_table(model_stats)
            print_formatted_table(df)

            # Optionally save to CSV
            output_file = "model_pass_rates.csv"
            df.to_csv(output_file, index=False)
            print(f"\nTable saved to: {output_file}")

        else:
            print("No test data found in the file.")

    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")
