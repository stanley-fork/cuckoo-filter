#!/usr/bin/env -s uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
#     "pandas"
# ]
# ///

import pandas as pd
import matplotlib.pyplot as plt
import sys


df = pd.read_csv(sys.stdin)

plt.figure(figsize=(12, 8))
colors = ["blue", "red", "green"]
for i, tableType in enumerate(df["tableType"].unique()):
    data = df[df["tableType"] == tableType]
    plt.loglog(
        data["n"],
        data["avgTimeMs"],
        marker="o",
        label=tableType,
        linewidth=2,
        color=colors[i % len(colors)],
    )

plt.xlabel("Input Size (n)")
plt.ylabel("Average Runtime (ms)")
plt.title("Hash Table Performance Comparison")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
