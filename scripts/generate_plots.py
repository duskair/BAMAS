"""
Generates two precise visualizations from a results JSONL log:
1. Overall Difficulty Histogram with Low/Med/High separation lines, ticks at every 0.1, and sample size.
2. A 10-panel grid of Topology Distribution by Budget, separated by 0.1 difficulty bins.

Usage:
    python scripts/generate_plots.py experiments/results/experiment_results_YYYYMMDD_HHMMSS.jsonl [DATASET_NAME]
Example:
    python scripts/generate_plots.py experiments/results/experiment_results_20260718_210520.jsonl GSM8K
"""
import json
import sys
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TOPOLOGY_COLORS = {
    "linear_topology": "#6EABF5",           # Blue
    "star_topology": "#9558B2",             # Purple
    "feedback_topology": "#28B463",         # Green
    "planner_driven_topology": "#8B2E2E"    # Dark Red
}

def load_results(path: Path) -> list[dict]:
    """Loads JSONL data and filters out rows without a difficulty score."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    missing = [r["work_id"] for r in rows if r.get("difficulty_score") is None]
    if missing:
        print(f"[!] Warning: {len(missing)} rows have no difficulty_score.")

    return [r for r in rows if r.get("difficulty_score") is not None]

def plot_histogram_1(rows: list[dict], out_path: Path, dataset_name: str):
    """
    Creates Histogram 1: Overall difficulty distribution with 1/3 and 2/3 cutoff lines,
    x-axis ticks at every 0.1, and total sample size shown.
    """
    scores = [r["difficulty_score"] for r in rows]
    n = len(scores)

    fig, ax = plt.subplots(figsize=(10, 6))

    n_bins = 20
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ax.hist(scores, bins=bin_edges, color="#BDC3C7", edgecolor="black", alpha=0.8)

    ax.axvline(x=1/3, color="#34495E", linestyle="--", linewidth=2.5, label="Low / Med Boundary (0.33)")
    ax.axvline(x=2/3, color="#34495E", linestyle=":", linewidth=2.5, label="Med / High Boundary (0.66)")

    max_y = ax.get_ylim()[1]
    if max_y == 0:
        max_y = 10  # Fallback if empty

    ax.text(1/6, max_y * 0.9, 'Low', horizontalalignment='center', fontsize=12, fontweight='bold')
    ax.text(0.5, max_y * 0.9, 'Medium', horizontalalignment='center', fontsize=12, fontweight='bold')
    ax.text(5/6, max_y * 0.9, 'High', horizontalalignment='center', fontsize=12, fontweight='bold')

    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlim(0, 1)
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    ax.set_title(f"Overall Task Difficulty Distribution - {dataset_name} (n={n} tasks)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Difficulty Score", fontsize=12)
    ax.set_ylabel("Number of Tasks", fontsize=12)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[*] Saved {out_path.name}")

def plot_histogram_2(rows: list[dict], out_path: Path, dataset_name: str):
    """
    Creates Histogram 2: A 1x3 grid of stacked bar charts for Low (<0.33), Medium (0.33-0.66), 
    and High (>0.66) difficulty bands.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    bands = [
        ("Low", 0.0, 0.33),
        ("Medium", 0.4, 0.66),
        ("High", 0.6, 1.0)
    ]
    budgets = sorted(list(set(r["budget"] for r in rows)))

    for idx, (band_name, lower, upper) in enumerate(bands):
        ax = axes[idx]

        if idx == 2:  
            bin_rows = [r for r in rows if lower <= r["difficulty_score"] <= upper]
        else:
            bin_rows = [r for r in rows if lower <= r["difficulty_score"] < upper]

        bottom = np.zeros(len(budgets))

        for topo, color in TOPOLOGY_COLORS.items():
            counts = []
            for b in budgets:
                count = sum(1 for r in bin_rows if r["budget"] == b and r["pattern_name"] == topo)
                counts.append(count)

            clean_label = topo.replace("_topology", "").replace("_driven", "").title()

            ax.bar([str(int(b)) for b in budgets], counts, bottom=bottom,
                   color=color, label=clean_label, edgecolor="white", width=0.5)
            bottom += np.array(counts)

        range_str = f"[{lower:.2f}, {upper:.2f})" if idx < 2 else f"[{lower:.2f}, {upper:.2f}]"
        ax.set_title(f"{band_name} Difficulty {range_str}\n(n={len(bin_rows)} tasks)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Nominal Budget", fontsize=11)

        if idx == 0:
            ax.set_ylabel("Number of Tasks", fontsize=11)

        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(TOPOLOGY_COLORS),
               bbox_to_anchor=(0.5, -0.05), fontsize=11, frameon=True, shadow=True)

    total_n = len(rows)
    fig.suptitle(f"Topology Distribution by Budget - {dataset_name} (n={total_n} tasks)\n(Segmented by Difficulty Bands)",
                 fontsize=15, fontweight="bold", y=1.03)

    fig.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Saved {out_path.name}")

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python scripts/generate_plots.py experiments/results/experiment_results_YYYYMMDD_HHMMSS.jsonl [DATASET_NAME]")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    dataset_name = sys.argv[2].upper() if len(sys.argv) == 3 else "Dataset"

    rows = load_results(in_path)
    if not rows:
        print("[!] No rows with difficulty_score found. Nothing to plot.")
        sys.exit(1)

    out_dir = in_path.parent

    match = re.search(r'(\d{8}_\d{6})', in_path.name)
    if match:
        timestamp = match.group(1)
    else:
        timestamp = in_path.stem.replace("experiment_results_", "")

    hist1_out = out_dir / f"difficulty_histogram_{dataset_name}_{timestamp}.png"
    hist2_out = out_dir / f"topology_by_budget_{dataset_name}_{timestamp}.png"

    plot_histogram_1(rows, hist1_out, dataset_name)
    plot_histogram_2(rows, hist2_out, dataset_name)

if __name__ == "__main__":
    main()