#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "typer",
# ]
# ///

import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

app = typer.Typer(help="Plot FPR sweep benchmark results")

FILTER_TYPES = {
    "GPUCF": "GPU Cuckoo",
    "CPUCF": "CPU Cuckoo",
    "Bloom": "Blocked Bloom",
    "TCF": "TCF",
    "GQF": "GQF",
    "PartitionedCF": "Partitioned CF",
}

FILTER_COLORS = {
    "GPU Cuckoo": "#2E86AB",
    "CPU Cuckoo": "#00B4D8",
    "Blocked Bloom": "#A23B72",
    "TCF": "#C73E1D",
    "GQF": "#F18F01",
    "Partitioned CF": "#6A994E",
}


def parse_benchmark_name(name: str) -> dict:
    """Extract filter type, fingerprint bits, load factor, and operation from benchmark name."""
    result = {
        "filter": None,
        "fingerprint_bits": None,
        "load_factor": None,
        "operation": "query",  # default to query for FPR_Sweep benchmarks
    }

    # Check if this is an insert benchmark
    if "_Insert_Sweep" in name:
        result["operation"] = "insert"

    for prefix, filter_name in FILTER_TYPES.items():
        if name.startswith(prefix):
            result["filter"] = filter_name
            break

    type_to_bits = {
        "uint8_t": 8,
        "uint16_t": 16,
        "uint32_t": 32,
        "uint64_t": 64,
    }

    # Extract template parameters
    if "<" in name and ">" in name:
        params = name[name.index("<") + 1 : name.index(">")].split(",")
        params = [p.strip() for p in params]

        if len(params) == 2:
            first_param = params[0]
            if first_param in type_to_bits:
                result["fingerprint_bits"] = type_to_bits[first_param]
            else:
                result["fingerprint_bits"] = int(first_param)
            result["load_factor"] = int(params[1]) / 100.0
        elif len(params) == 1:
            result["load_factor"] = int(params[0]) / 100.0

    return result


def create_filter_comparison_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Create 2 files: fastest filter heatmap, and insert/query throughput heatmaps."""

    # Define bins for FPR and bits_per_item
    fpr_bins = np.array([2**i for i in range(-16, 0)])
    space_bins = np.array([4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 50, 64, 100])

    n_fpr = len(fpr_bins) - 1
    n_space = len(space_bins) - 1

    filter_names = list(FILTER_COLORS.keys())
    filter_to_idx = {name: i for i, name in enumerate(filter_names)}

    # Initialize grids
    fastest_filter_avg = np.full((n_fpr, n_space), -1, dtype=int)
    best_avg_throughput = np.zeros((n_fpr, n_space))
    best_insert_throughput = np.zeros((n_fpr, n_space))
    best_query_throughput = np.zeros((n_fpr, n_space))

    # Separate insert and query data
    query_df = df[df["operation"] == "query"].copy()
    insert_df = df[df["operation"] == "insert"].copy()

    # Create config key for matching insert/query pairs
    def make_config_key(row):
        return f"{row['filter']}_{row.get('fingerprint_bits', '')}_{row.get('load_factor', '')}"

    query_df["config_key"] = query_df.apply(make_config_key, axis=1)
    insert_df["config_key"] = insert_df.apply(make_config_key, axis=1)

    # Merge insert and query data
    merged = pd.merge(
        query_df[
            [
                "config_key",
                "filter",
                "fpr_percentage",
                "bits_per_item",
                "throughput_mops",
            ]
        ],
        insert_df[["config_key", "throughput_mops"]],
        on="config_key",
        suffixes=("_query", "_insert"),
        how="outer",
    )

    # Fill NaN filter from query or insert data
    if "filter" not in merged.columns or merged["filter"].isna().any():
        insert_filter_map = insert_df.set_index("config_key")["filter"].to_dict()
        merged["filter"] = merged.apply(
            lambda r: r["filter"]
            if pd.notna(r.get("filter"))
            else insert_filter_map.get(r["config_key"]),
            axis=1,
        )

    # Calculate average throughput
    q_tp = merged["throughput_mops_query"].fillna(0)
    i_tp = merged["throughput_mops_insert"].fillna(0)

    both_exist = (merged["throughput_mops_query"].notna()) & (
        merged["throughput_mops_insert"].notna()
    )
    merged["avg_throughput"] = np.where(both_exist, (q_tp + i_tp) / 2, q_tp + i_tp)

    # Assign each data point to bins
    for _, row in merged.iterrows():
        filter_name = row.get("filter")
        if pd.isna(filter_name) or filter_name not in filter_to_idx:
            continue

        fpr = row.get("fpr_percentage", 0)
        if pd.isna(fpr) or fpr <= 0:
            continue
        fpr = fpr / 100

        bits = row.get("bits_per_item", 0)
        if pd.isna(bits) or bits <= 0:
            continue

        avg_tp = row.get("avg_throughput", 0) or 0
        query_tp = row.get("throughput_mops_query", 0) or 0
        insert_tp = row.get("throughput_mops_insert", 0) or 0

        fpr_idx = np.searchsorted(fpr_bins, fpr) - 1
        space_idx = np.searchsorted(space_bins, bits) - 1

        if 0 <= fpr_idx < n_fpr and 0 <= space_idx < n_space:
            if avg_tp > best_avg_throughput[fpr_idx, space_idx]:
                best_avg_throughput[fpr_idx, space_idx] = avg_tp
                fastest_filter_avg[fpr_idx, space_idx] = filter_to_idx[filter_name]

            if query_tp > best_query_throughput[fpr_idx, space_idx]:
                best_query_throughput[fpr_idx, space_idx] = query_tp

            if insert_tp > best_insert_throughput[fpr_idx, space_idx]:
                best_insert_throughput[fpr_idx, space_idx] = insert_tp

    colors = ["white"] + [FILTER_COLORS[name] for name in filter_names]
    cmap = ListedColormap(colors)

    # File 1: Fastest filter heatmap
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.imshow(
        fastest_filter_avg + 1, cmap=cmap, aspect="auto", vmin=0, vmax=len(filter_names)
    )
    ax1.set_xticks(range(n_space))
    ax1.set_xticklabels([f"{int(space_bins[i])}" for i in range(n_space)], fontsize=9)
    ax1.set_yticks(range(n_fpr))
    ax1.set_yticklabels([f"$2^{{{int(np.log2(fpr_bins[i]))}}}$" for i in range(n_fpr)])
    ax1.set_xlabel("Bits per item")
    ax1.set_ylabel("FPR")
    ax1.set_title("Fastest Filter")
    legend_elements = [
        Patch(facecolor=FILTER_COLORS[name], label=name) for name in filter_names
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)
    plt.tight_layout()

    fastest_path = output_dir / "fpr_sweep_fastest.png"
    plt.savefig(fastest_path, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    typer.secho(f"Saved fastest filter plot to {fastest_path}", fg=typer.colors.GREEN)

    # File 2: Insert and Query throughput heatmaps
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Insert throughput
    ax2 = axes[0]
    insert_masked = np.ma.masked_where(
        best_insert_throughput == 0, best_insert_throughput
    )
    if insert_masked.count() > 0 and insert_masked.max() > 0:
        im2 = ax2.imshow(
            insert_masked,
            cmap="viridis",
            aspect="auto",
            norm=plt.matplotlib.colors.LogNorm(
                vmin=max(1, insert_masked.min()), vmax=insert_masked.max()
            ),
        )
        plt.colorbar(im2, ax=ax2, label="MOPS")
    else:
        ax2.text(
            0.5,
            0.5,
            "No insert data",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
    ax2.set_xticks(range(n_space))
    ax2.set_xticklabels([f"{int(space_bins[i])}" for i in range(n_space)], fontsize=8)
    ax2.set_yticks(range(n_fpr))
    ax2.set_yticklabels([f"$2^{{{int(np.log2(fpr_bins[i]))}}}$" for i in range(n_fpr)])
    ax2.set_xlabel("Bits per item")
    ax2.set_ylabel("FPR")
    ax2.set_title("Insert Throughput (MOPS)")

    # Query throughput
    ax3 = axes[1]
    query_masked = np.ma.masked_where(best_query_throughput == 0, best_query_throughput)
    if query_masked.count() > 0 and query_masked.max() > 0:
        im3 = ax3.imshow(
            query_masked,
            cmap="viridis",
            aspect="auto",
            norm=plt.matplotlib.colors.LogNorm(
                vmin=max(1, query_masked.min()), vmax=query_masked.max()
            ),
        )
        plt.colorbar(im3, ax=ax3, label="MOPS")
    ax3.set_xticks(range(n_space))
    ax3.set_xticklabels([f"{int(space_bins[i])}" for i in range(n_space)], fontsize=8)
    ax3.set_yticks(range(n_fpr))
    ax3.set_yticklabels([f"$2^{{{int(np.log2(fpr_bins[i]))}}}$" for i in range(n_fpr)])
    ax3.set_xlabel("Bits per item")
    ax3.set_ylabel("FPR")
    ax3.set_title("Query Throughput (MOPS)")

    plt.tight_layout()

    throughput_path = output_dir / "fpr_sweep_throughput.png"
    plt.savefig(throughput_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    typer.secho(f"Saved throughput plots to {throughput_path}", fg=typer.colors.GREEN)


@app.command()
def main(
    csv_file: Path = typer.Argument(
        "-",
        help="Path to benchmark CSV file, or '-' to read from stdin (default: stdin)",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory for plots (default: build/)",
    ),
):
    """
    Plot FPR sweep benchmark results, generating two files:
    1. fpr_sweep_fastest.png - Fastest filter heatmap (by avg insert+query throughput)
    2. fpr_sweep_throughput.png - Insert and Query throughput heatmaps
    """
    try:
        if str(csv_file) == "-":
            df = pd.read_csv(sys.stdin)
        elif not csv_file.exists():
            typer.secho(
                f"Error: File not found: {csv_file}", fg=typer.colors.RED, err=True
            )
            raise typer.Exit(1)
        else:
            df = pd.read_csv(csv_file)
    except Exception as e:
        typer.secho(f"Error parsing CSV: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    df = df[df["name"].str.contains("_median", na=False)]

    parsed = df["name"].apply(parse_benchmark_name)
    df["filter"] = parsed.apply(lambda x: x["filter"])
    df["fingerprint_bits"] = parsed.apply(lambda x: x["fingerprint_bits"])
    df["load_factor"] = parsed.apply(lambda x: x["load_factor"])
    df["operation"] = parsed.apply(lambda x: x["operation"])

    df["throughput_mops"] = df["items_per_second"] / 1e6

    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "build"

    output_dir.mkdir(parents=True, exist_ok=True)

    create_filter_comparison_heatmaps(df, output_dir)


if __name__ == "__main__":
    app()
