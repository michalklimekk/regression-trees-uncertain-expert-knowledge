"""Sensitivity analysis for alternative expert-confidence mappings."""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from simulate_larger_trees import SCENARIOS, run_repetition


SCENARIO_NAMES = ("uncertain_panel", "confidently_wrong")
CONFIDENCE_METHODS = ("entropy", "margin", "coverage")


def summarize(df):
    rows = []
    for scenario in SCENARIO_NAMES:
        for method in CONFIDENCE_METHODS:
            block = df[
                (df["scenario"] == scenario)
                & (df["confidence_method"] == method)
            ]
            pivot = block.pivot(index="repetition", columns="variant", values="rmse")
            for variant in ["posthoc", "full"]:
                diff = (pivot[variant] - pivot["cart"]).to_numpy()
                mean = float(np.mean(diff))
                se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
                crit = float(t.ppf(0.975, len(diff) - 1))
                rows.append(
                    {
                        "scenario": scenario,
                        "confidence_method": method,
                        "variant": variant,
                        "difference_vs_cart": mean,
                        "ci95_low": mean - crit * se,
                        "ci95_high": mean + crit * se,
                    }
                )
    return pd.DataFrame(rows)


def plot_results(summary, figures):
    labels = ["Entropy", "Margin", "Coverage"]
    methods = list(CONFIDENCE_METHODS)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(methods))
    offsets = {
        ("uncertain_panel", "posthoc"): -0.18,
        ("uncertain_panel", "full"): -0.06,
        ("confidently_wrong", "posthoc"): 0.06,
        ("confidently_wrong", "full"): 0.18,
    }
    for scenario, variant in offsets:
        block = (
            summary[
                (summary["scenario"] == scenario)
                & (summary["variant"] == variant)
            ]
            .set_index("confidence_method")
            .loc[methods]
        )
        y = block["difference_vs_cart"].to_numpy()
        err = np.vstack(
            [
                y - block["ci95_low"].to_numpy(),
                block["ci95_high"].to_numpy() - y,
            ]
        )
        label = (
            f"{scenario.replace('_', ' ')}: "
            f"{'post-hoc' if variant == 'posthoc' else 'full'}"
        )
        ax.errorbar(
            x + offsets[(scenario, variant)],
            y,
            yerr=err,
            marker="o",
            capsize=3,
            linestyle="none",
            label=label,
        )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE difference vs CART")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "confidence_mapping_sensitivity.png", dpi=220)
    fig.savefig(figures / "confidence_mapping_sensitivity.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    selected = {s.name: s for s in SCENARIOS if s.name in SCENARIO_NAMES}
    master = np.random.default_rng(args.seed)
    rows = []
    for scenario in selected.values():
        seeds = [
            int(master.integers(0, 2**31 - 1)) for _ in range(args.repetitions)
        ]
        for method in CONFIDENCE_METHODS:
            for repetition, seed in enumerate(seeds):
                block = run_repetition(
                    scenario,
                    repetition,
                    seed,
                    confidence_method=method,
                )
                for row in block:
                    row["confidence_method"] = method
                rows.extend(block)

    df = pd.DataFrame(rows)
    summary = summarize(df)
    df.to_csv(args.results / "confidence_sensitivity_repetitions.csv", index=False)
    summary.to_csv(args.results / "confidence_sensitivity_summary.csv", index=False)
    plot_results(summary, args.figures)


if __name__ == "__main__":
    main()
