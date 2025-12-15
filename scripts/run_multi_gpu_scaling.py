#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "typer",
# ]
# ///

import os
import subprocess
import tempfile
from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def main(
    output: Path = typer.Option(
        Path("build/multi_gpu_scaling.csv"),
        "--output",
        "-o",
        help="Output CSV file path",
    ),
    gpu_counts: str = typer.Option(
        "2,4,6,8",
        "--gpu-counts",
        "-g",
        help="Comma-separated list of GPU counts to test",
    ),
):
    """Run multi-GPU scaling benchmark with varying GPU counts.

    Uses CUDA_VISIBLE_DEVICES to control which GPUs are used for each run.
    Combines results into a single CSV file.
    """
    build_dir = Path(__file__).parent.parent / "build"
    benchmark_exe = build_dir / "benchmark-multi-gpu-scaling"

    if not benchmark_exe.exists():
        typer.secho(
            f"Error: {benchmark_exe} not found. Did you run 'meson compile -C build'?",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)

    counts = [int(c.strip()) for c in gpu_counts.split(",")]

    all_lines = []
    header = None

    for num_gpus in counts:
        # Create GPU list: 0,1 for 2 GPUs, 0,1,2,3 for 4 GPUs, etc.
        gpu_list = ",".join(str(i) for i in range(num_gpus))

        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Running with {num_gpus} GPUs (CUDA_VISIBLE_DEVICES={gpu_list})")
        typer.echo(f"{'=' * 60}")

        # Create temporary file for CSV output
        with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as f:
            csv_path = f.name

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list

        result = subprocess.run(
            [
                str(benchmark_exe),
                f"--benchmark_out={csv_path}",
                "--benchmark_out_format=csv",
                "--benchmark_format=csv",
            ],
            env=env,
        )

        if result.returncode != 0:
            typer.secho(
                f"Error running benchmark with {num_gpus} GPUs",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(1)

        # Read CSV from temp file
        with open(csv_path) as f:
            lines = [line.rstrip() for line in f if line.strip()]

        Path(csv_path).unlink()  # Clean up temp file

        if not lines:
            typer.secho(
                f"Warning: No output from {num_gpus} GPU run",
                fg=typer.colors.YELLOW,
            )
            continue

        if header is None:
            header = lines[0]
            all_lines.append(header)
            all_lines.extend(lines[1:])
        else:
            # Skip header for subsequent runs
            if lines[0] == header:
                all_lines.extend(lines[1:])
            else:
                all_lines.extend(lines)

    # Write combined CSV
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        for line in all_lines:
            f.write(line + "\n")

    typer.secho(f"\nResults written to {output}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
