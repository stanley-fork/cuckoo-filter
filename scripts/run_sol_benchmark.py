#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "typer",
# ]
# ///

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from ncu_profiler import FILTERS, OPERATIONS, run_ncu_profile, to_superscript

app = typer.Typer(
    help="Run Speed of Light (SOL) throughput benchmarks with NCU profiling"
)

DEFAULT_MIN_CAPACITY_LOG2 = 16
DEFAULT_MAX_CAPACITY_LOG2 = 28

SOL_METRICS_MAP = {
    "sm_throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "memory_throughput": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1_throughput": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l2_throughput": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_throughput": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
}

SOL_METRICS_LIST = list(SOL_METRICS_MAP.values())


@app.command()
def main(
    executable: Path = typer.Argument(
        ...,
        help="Path to the cache_benchmark executable",
        exists=True,
    ),
    output: Path = typer.Option(
        "sol_benchmark_results.csv",
        "--output",
        "-o",
        help="Output CSV file for results",
    ),
    min_capacity_log2: int = typer.Option(
        DEFAULT_MIN_CAPACITY_LOG2,
        "--min-log2",
        help="Minimum capacity as log2(capacity)",
    ),
    max_capacity_log2: int = typer.Option(
        DEFAULT_MAX_CAPACITY_LOG2,
        "--max-log2",
        help="Maximum capacity as log2(capacity)",
    ),
    load_factor: float = typer.Option(
        0.95,
        "--load-factor",
        "-l",
        help="Load factor for all runs",
    ),
    filters: Optional[list[str]] = typer.Option(
        None,
        "--filter",
        "-f",
        help="Specific filters to test (can specify multiple)",
    ),
):
    """
    Profile Speed of Light (SOL) throughputs across varying input sizes.

    Measures the following throughputs as % of peak:
    - Compute (SM)
    - Memory (Overall)
    - L1/TEX Cache
    - L2 Cache
    - DRAM
    """
    if not executable.exists():
        typer.secho(
            f"Executable not found: {executable}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    test_filters = filters if filters else FILTERS

    exponents = list(range(min_capacity_log2, max_capacity_log2 + 1))
    capacities = [2**e for e in exponents]

    typer.secho(
        f"Testing {len(test_filters)} filter(s) across {len(exponents)} capacity values with load factor {load_factor}",
        fg=typer.colors.GREEN,
        err=True,
    )
    typer.secho(
        f"Capacity range: 2{to_superscript(exponents[0])} ({capacities[0]:,}) to 2{to_superscript(exponents[-1])} ({capacities[-1]:,})",
        fg=typer.colors.GREEN,
        err=True,
    )

    results = []
    total_runs = sum(len(OPERATIONS[f]) for f in test_filters) * len(exponents)
    current_run = 0

    for filter_type in test_filters:
        for operation in OPERATIONS[filter_type]:
            for exponent in exponents:
                current_run += 1
                typer.secho(
                    f"\n[{current_run}/{total_runs}] ",
                    fg=typer.colors.BRIGHT_BLUE,
                    err=True,
                    nl=False,
                )

                metrics = run_ncu_profile(
                    executable,
                    filter_type,
                    operation,
                    exponent,
                    load_factor,
                    SOL_METRICS_LIST,
                )

                if metrics:
                    capacity = 2**exponent

                    result_entry = {
                        "filter": filter_type,
                        "operation": operation,
                        "capacity": capacity,
                        "load_factor": load_factor,
                    }

                    # Extract metrics using the map
                    log_parts = []
                    for friendly_name, ncu_name in SOL_METRICS_MAP.items():
                        value = metrics.get(ncu_name, 0.0)
                        result_entry[friendly_name] = value
                        log_parts.append(
                            f"{friendly_name.split('_')[0].upper()}: {value:.1f}%"
                        )

                    results.append(result_entry)

                    typer.secho(
                        "  " + ", ".join(log_parts),
                        fg=typer.colors.GREEN,
                        err=True,
                    )

    if not results:
        typer.secho(
            "\nNo results collected",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    df = pd.DataFrame(results)
    df.to_csv(output, index=False)

    typer.secho(
        f"\nResults saved to {output} ({len(results)} data points)",
        fg=typer.colors.GREEN,
        err=True,
    )

    typer.secho("\nSummary Statistics (Mean %):", fg=typer.colors.BRIGHT_CYAN, err=True)
    summary_cols = list(SOL_METRICS_MAP.keys())
    summary = df.groupby(["filter", "operation"])[summary_cols].mean().round(2)
    print(summary)


if __name__ == "__main__":
    app()
