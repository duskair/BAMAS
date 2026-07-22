"""
Compares two experiment runs (difficulty ON vs OFF)

Difficulty only restricts the model tier (easy tasks -> cheap model); The figure has:
    1. Accuracy by budget
    2. Average actual cost by budget
    3. Planning feasibility by budget
    4. Cost vs accuracy tradeoff

FIRST file is the difficulty ON run (diffon) and SECOND is the baseline (diffoff).

Usage:
    python scripts/compare.py DIFFON.jsonl DIFFOFF.jsonl [DATASET] [--labels "A" "B"] [--out DIR]

Example:
    python scripts/compare.py experiments/results/experiment_results_diffon_20260721_184254.jsonl experiments/results/experiment_results_diffoff_20260721_185429.jsonl GSM8K
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLOR_ON = "#2E86C1"    # difficulty ON
COLOR_OFF = "#E67E22"   # difficulty OFF (baseline)


def load_results(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def budgets_of(rows: list[dict]) -> list[float]:
    return sorted({r["budget"] for r in rows})


def _rate(rows: list[dict], budget: float, key: str) -> float:
    sub = [r for r in rows if r["budget"] == budget]
    return sum(1 for r in sub if r.get(key)) / len(sub) if sub else 0.0


def _avg(rows: list[dict], budget: float, key: str) -> float:
    sub = [r for r in rows if r["budget"] == budget and r.get(key) is not None]
    return float(np.mean([r[key] for r in sub])) if sub else 0.0


def _overall_rate(rows: list[dict], key: str) -> float:
    return sum(1 for r in rows if r.get(key)) / len(rows) if rows else 0.0


def _grouped(ax, budgets, v_on, v_off, label_on, label_off, ylabel, title, as_pct=False, annotate_pct_change=False):
    x = np.arange(len(budgets))
    w = 0.38
    s = 100.0 if as_pct else 1.0
    b_on = ax.bar(x - w / 2, [v * s for v in v_on], w, label=label_on, color=COLOR_ON, edgecolor="white")
    b_off = ax.bar(x + w / 2, [v * s for v in v_off], w, label=label_off, color=COLOR_OFF, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels([str(int(b)) for b in budgets])
    ax.set_xlabel("Budget", fontsize=11); ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4); ax.set_axisbelow(True); ax.legend()
    for bars in (b_on, b_off):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}", (bar.get_x() + bar.get_width() / 2, h),
                        textcoords="offset points", xytext=(0, 2), ha="center", fontsize=8)
    if annotate_pct_change:
        for i, (a, b) in enumerate(zip(v_on, v_off)):
            if b:
                pct = (a - b) / b * 100.0
                ax.annotate(f"{pct:+.0f}%", (x[i], max(a, b) * s),
                            textcoords="offset points", xytext=(0, 14), ha="center",
                            fontsize=9, fontweight="bold",
                            color="#1E8449" if pct < 0 else "#922B21")


def plot_comparison(on_rows, off_rows, label_on, label_off, out_path: Path, dataset: str):
    budgets = sorted(set(budgets_of(on_rows)) | set(budgets_of(off_rows)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    _grouped(axes[0, 0], budgets,
             [_rate(on_rows, b, "is_correct") for b in budgets],
             [_rate(off_rows, b, "is_correct") for b in budgets],
             label_on, label_off, "Accuracy (%)", "Accuracy by budget", as_pct=True)

    _grouped(axes[0, 1], budgets,
             [_avg(on_rows, b, "actual_cost") for b in budgets],
             [_avg(off_rows, b, "actual_cost") for b in budgets],
             label_on, label_off, "Average actual cost", "Cost by budget",
             as_pct=False, annotate_pct_change=True)

    _grouped(axes[1, 0], budgets,
             [_rate(on_rows, b, "planning_feasible") for b in budgets],
             [_rate(off_rows, b, "planning_feasible") for b in budgets],
             label_on, label_off, "Planning feasible (%)", "Feasibility by budget", as_pct=True)

    ax = axes[1, 1]
    for rows, color, lbl in [(on_rows, COLOR_ON, label_on), (off_rows, COLOR_OFF, label_off)]:
        xs = [_avg(rows, b, "actual_cost") for b in budgets]
        ys = [_rate(rows, b, "is_correct") * 100 for b in budgets]
        order = np.argsort(xs)
        xs = np.array(xs)[order]; ys = np.array(ys)[order]
        ax.plot(xs, ys, "-o", color=color, label=lbl, markersize=7, linewidth=1.5)
    ax.set_xlabel("Average actual cost", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Cost vs accuracy tradeoff\n(upper-left = cheaper & more accurate)", fontsize=12, fontweight="bold")
    ax.grid(linestyle="--", alpha=0.4); ax.set_axisbelow(True); ax.legend()

    c_on, c_off = float(np.mean([r["actual_cost"] for r in on_rows])), float(np.mean([r["actual_cost"] for r in off_rows]))
    a_on, a_off = _overall_rate(on_rows, "is_correct"), _overall_rate(off_rows, "is_correct")
    cost_chg = (c_on - c_off) / c_off * 100 if c_off else 0.0
    fig.suptitle(f"{label_on}  vs  {label_off} -- {dataset}\n"
                 f"cost {c_on:.0f} vs {c_off:.0f} ({cost_chg:+.0f}%)   |   accuracy {a_on:.2f} vs {a_off:.2f}",
                 fontsize=16, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[*] Saved {out_path.name}")


def print_summary(on_rows, off_rows, label_on, label_off, dataset: str):
    c_on = float(np.mean([r["actual_cost"] for r in on_rows])) if on_rows else 0.0
    c_off = float(np.mean([r["actual_cost"] for r in off_rows])) if off_rows else 0.0
    print("\n" + "=" * 74)
    print(f"COMPARISON - {dataset}")
    print("=" * 74)
    print(f"{'Metric':<26}{label_on:>22}{label_off:>22}")
    print("-" * 74)
    print(f"{'Rows':<26}{len(on_rows):>22}{len(off_rows):>22}")
    print(f"{'Overall accuracy':<26}{_overall_rate(on_rows,'is_correct'):>22.3f}{_overall_rate(off_rows,'is_correct'):>22.3f}")
    print(f"{'Planning feasible rate':<26}{_overall_rate(on_rows,'planning_feasible'):>22.3f}{_overall_rate(off_rows,'planning_feasible'):>22.3f}")
    print(f"{'Avg actual cost':<26}{c_on:>22.2f}{c_off:>22.2f}")
    if c_off:
        print(f"{'Cost change (ON vs OFF)':<26}{(c_on-c_off)/c_off*100:>21.1f}%{'':>22}")
    print("=" * 74)


def extract_timestamp(path: Path) -> str:
    """Extracts timestamp (YYYYMMDD_HHMMSS) from filename or falls back to current time."""
    match = re.search(r'(\d{8}_\d{6})', path.name)
    if match:
        return match.group(1)
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser(description="Compare a difficulty-ON run against a difficulty-OFF baseline.")
    ap.add_argument("diffon_file", type=str, help="Results JSONL for the difficulty-ON run.")
    ap.add_argument("diffoff_file", type=str, help="Results JSONL for the difficulty-OFF baseline.")
    ap.add_argument("dataset", nargs="?", default="Dataset", help="Dataset name for titles.")
    ap.add_argument("--labels", nargs=2, metavar=("ON", "OFF"), default=("difficulty ON", "difficulty OFF"),
                    help="Override the two run labels (default: 'difficulty ON' / 'difficulty OFF').")
    ap.add_argument("--out", type=str, default=None, help="Output directory (default: alongside the diffon file).")
    args = ap.parse_args()

    on_path, off_path = Path(args.diffon_file), Path(args.diffoff_file)
    dataset = args.dataset.upper()
    label_on, label_off = args.labels
    on_rows, off_rows = load_results(on_path), load_results(off_path)
    if not on_rows or not off_rows:
        print("[!] One of the input files is empty. Nothing to compare.")
        return

    out_dir = Path(args.out) if args.out else on_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = extract_timestamp(on_path)
    out_filename = f"compare_difficulty_{dataset}_{timestamp}.png"

    print_summary(on_rows, off_rows, label_on, label_off, dataset)
    plot_comparison(on_rows, off_rows, label_on, label_off, out_dir / out_filename, dataset)


if __name__ == "__main__":
    main()