#!/usr/bin/env python3
"""Generate paper figures from independent-process benchmark summaries."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REPO_ROOT / "target" / "release" / "bench_output_final"
OUTPUT_DIR = Path(__file__).resolve().parent / "media"
PROCESS_COUNT = 3
T_CRITICAL_95_DF2 = 4.302652729911275

COLORS = {
    "Dense reference": "#4C566A",
    "Full ACHF (guarded AMA)": "#D1495B",
    "Sparse training (fixed mask)": "#2A9D8F",
    "Static magnitude pruning": "#E9A23B",
    "Guarded AMA": "#0072B2",
    "Plain EMA": "#D55E00",
    "Cached": "#0072B2",
    "Sparse": "#009E73",
    "Dense": "#D55E00",
}

DISPLAY_LABELS = {
    "Full ACHF (guarded AMA)": "ACHF runtime\n(gate inactive)",
    "Cached": "Prepared Candidate",
    "Sparse": "Candidate CSR",
    "Dense": "Candidate Dense",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def process_file(index: int, name: str) -> Path:
    return BENCH_ROOT / f"process_{index:03d}" / name


def mean_ci(values: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    if len(samples) != PROCESS_COUNT:
        raise ValueError(f"expected {PROCESS_COUNT} process means, got {len(samples)}")
    sample_mean = statistics.fmean(samples)
    sample_std = statistics.stdev(samples)
    margin = T_CRITICAL_95_DF2 * sample_std / math.sqrt(len(samples))
    return {
        "mean": sample_mean,
        "ci_95": [sample_mean - margin, sample_mean + margin],
        "process_means": samples,
    }


def asymmetric_error(stats: dict[str, Any]) -> list[list[float]]:
    mean = stats["mean"]
    low, high = stats["ci_95"]
    return [[mean - low], [high - mean]]


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "svg", "png"):
        options: dict[str, Any] = {"bbox_inches": "tight"}
        if extension == "png":
            options["dpi"] = 300
        output_path = OUTPUT_DIR / f"{stem}.{extension}"
        fig.savefig(output_path, **options)
        if extension == "svg":
            lines = output_path.read_text(encoding="utf-8").splitlines()
            output_path.write_text(
                "\n".join(line.rstrip() for line in lines) + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "talos-xii-achf-paper",
        }
    )


def plot_ablation(derived: dict[str, Any]) -> None:
    aggregate = load_json(BENCH_ROOT / "summary.json")
    rows = aggregate["experiments"]["ablation"]
    order = [
        "Dense reference",
        "Full ACHF (guarded AMA)",
        "Sparse training (fixed mask)",
        "Static magnitude pruning",
    ]
    by_label = {row["label"]: row for row in rows}
    y_positions = list(range(len(order)))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.05), constrained_layout=True)
    metrics = [
        ("throughput_sims_per_sec", "Throughput (simulations/s)"),
        ("eval_reward", "Evaluation reward"),
    ]
    figure_data: dict[str, Any] = {}
    for axis, (metric, label) in zip(axes, metrics):
        figure_data[metric] = {}
        for y, condition in zip(y_positions, order):
            stats = by_label[condition][metric]
            figure_data[metric][condition] = stats
            axis.errorbar(
                stats["mean"],
                y,
                xerr=asymmetric_error(stats),
                fmt="o",
                markersize=5,
                capsize=3,
                color=COLORS[condition],
                linewidth=1.4,
            )
        tick_labels = [DISPLAY_LABELS.get(condition, condition) for condition in order]
        axis.set_yticks(
            y_positions,
            tick_labels if axis is axes[0] else [""] * len(order),
        )
        axis.set_xlabel(label)
        axis.grid(axis="y", visible=False)
    axes[0].set_title("Runtime")
    axes[1].set_title("Task quality")
    fig.suptitle("ACHF ablation: mean and 95% CI over three process means", y=1.03)
    derived["ablation"] = figure_data
    save_figure(fig, "ablation_aggregate")


def plot_regime(derived: dict[str, Any]) -> None:
    process_rows = [
        load_json(process_file(index, "regime_adaptation_summary.json"))
        for index in range(1, PROCESS_COUNT + 1)
    ]
    batches = (1, 128)
    series_specs = (
        ("Guarded AMA", "guarded_ama_oracle_gap"),
        ("Plain EMA", "plain_ema_oracle_gap"),
    )
    figure_data: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.15), constrained_layout=True)

    for axis, batch in zip(axes, batches):
        reference_rows = sorted(
            (row for row in process_rows[0] if row["batch"] == batch),
            key=lambda row: row["actual_sparsity"],
        )
        x_values = [row["actual_sparsity"] for row in reference_rows]
        figure_data[str(batch)] = {}
        for label, field in series_specs:
            points = []
            for x_value in x_values:
                process_means = []
                for rows in process_rows:
                    row = next(
                        row
                        for row in rows
                        if row["batch"] == batch
                        and math.isclose(row["actual_sparsity"], x_value, abs_tol=1e-12)
                    )
                    process_means.append(row[field]["mean"])
                points.append(mean_ci(process_means))
            means = [point["mean"] for point in points]
            lower = [point["mean"] - point["ci_95"][0] for point in points]
            upper = [point["ci_95"][1] - point["mean"] for point in points]
            axis.errorbar(
                x_values,
                means,
                yerr=[lower, upper],
                marker="o",
                markersize=4,
                capsize=3,
                linewidth=1.6,
                color=COLORS[label],
                label=label,
            )
            figure_data[str(batch)][label] = [
                {"actual_sparsity": x_value, **point}
                for x_value, point in zip(x_values, points)
            ]
        axis.axhline(1.0, color="#666666", linestyle="--", linewidth=0.9, label="Oracle")
        axis.set_title(f"Batch {batch}")
        axis.set_xlabel("Actual weight sparsity")
        axis.set_xlim(min(x_values) - 0.005, max(x_values) + 0.005)
        axis.set_ylim(bottom=0.92)
        axis.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("Selector / oracle latency")
    fig.suptitle("Guarded versus plain selectors at warmed operating points", y=1.03)
    derived["regime"] = figure_data
    save_figure(fig, "regime_aggregate")


def plot_crossover(derived: dict[str, Any]) -> None:
    process_rows = [
        load_json(process_file(index, "path_crossover_summary.json"))
        for index in range(1, PROCESS_COUNT + 1)
    ]
    dimensions = (256, 1024, 2048)
    paths = ("Cached", "Sparse", "Dense")
    figure_data: dict[str, Any] = {}
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.35), sharey=True)

    for axis, dimension in zip(axes, dimensions):
        reference_rows = sorted(
            (row for row in process_rows[0] if row["dim"] == dimension),
            key=lambda row: row["actual_sparsity"],
        )
        x_values = [row["actual_sparsity"] for row in reference_rows]
        figure_data[str(dimension)] = {}
        for path in paths:
            field = f"{path.lower()}_ns"
            points = []
            for x_value in x_values:
                process_means_ms = []
                for rows in process_rows:
                    row = next(
                        row
                        for row in rows
                        if row["dim"] == dimension
                        and math.isclose(row["actual_sparsity"], x_value, abs_tol=1e-12)
                    )
                    process_means_ms.append(row[field]["mean"] / 1_000_000.0)
                points.append(mean_ci(process_means_ms))
            means = [point["mean"] for point in points]
            lower = [
                point["mean"] - max(point["ci_95"][0], point["mean"] * 0.02)
                for point in points
            ]
            upper = [point["ci_95"][1] - point["mean"] for point in points]
            axis.errorbar(
                x_values,
                means,
                yerr=[lower, upper],
                marker="o",
                markersize=3.8,
                capsize=2.5,
                linewidth=1.5,
                color=COLORS[path],
                label=DISPLAY_LABELS[path],
            )
            figure_data[str(dimension)][path] = [
                {"actual_sparsity": x_value, **point}
                for x_value, point in zip(x_values, points)
            ]
        axis.set_yscale("log")
        axis.set_title(f"Dimension {dimension}")
        axis.set_xlabel("Actual weight sparsity")
        axis.grid(which="minor", alpha=0.10)
    axes[0].set_ylabel("Path latency (ms, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle("Fixed-path crossover by dimension and sparsity", y=0.90)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.82))
    derived["crossover"] = figure_data
    save_figure(fig, "crossover_aggregate")


def main() -> None:
    if not (BENCH_ROOT / "summary.json").is_file():
        raise SystemExit(f"missing aggregate benchmark output: {BENCH_ROOT}")
    configure_style()
    derived: dict[str, Any] = {
        "source": str(BENCH_ROOT.relative_to(REPO_ROOT)).replace("\\", "/"),
        "statistical_unit": "independent process mean",
        "processes": PROCESS_COUNT,
        "confidence_interval": "two-sided Student-t, 95%, df=2",
    }
    plot_ablation(derived)
    plot_regime(derived)
    plot_crossover(derived)
    with (OUTPUT_DIR / "aggregate_figure_data.json").open("w", encoding="utf-8") as handle:
        json.dump(derived, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
