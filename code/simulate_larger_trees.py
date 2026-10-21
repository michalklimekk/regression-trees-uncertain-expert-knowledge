"""Controlled larger-tree simulations for uncertain directional knowledge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from simulation_utils import (
    generate_tree_regression,
    response_standardization,
    rmse,
    simulate_expert_panel,
    tune_lambda,
)
from uetree import UncertainExpertRegressionTree, aggregate_expert_votes, soft_calibrate


TRUE_DIRECTIONS = np.array([1, 1, 1, -1, 0, 0], dtype=int)


@dataclass(frozen=True)
class Scenario:
    name: str
    panel_center: str
    error_rate: float
    abstention_rate: float
    local_violation: float


SCENARIOS = [
    Scenario("aligned", "truth", 0.10, 0.05, 0.00),
    Scenario("local_violation", "truth", 0.10, 0.05, 0.85),
    Scenario("uncertain_panel", "truth", 0.35, 0.25, 0.40),
    Scenario("confidently_wrong", "reversed", 0.10, 0.05, 0.00),
]


def make_model(split_lambda=0.0, leaf_lambda=0.0, hard=False):
    return UncertainExpertRegressionTree(
        max_depth=4,
        min_samples_leaf=15,
        max_thresholds=32,
        split_lambda=split_lambda,
        leaf_lambda=leaf_lambda,
        hard_order=hard,
    )


def fit_full(X_train, y_train, X_val, y_val, directions, confidences):
    """Tune the common split/leaf penalty on the validation sample."""

    def evaluator(lam):
        model = make_model(lam, lam).fit(X_train, y_train, directions, confidences)
        return rmse(y_val, model.predict(X_val)), model

    best_lambda, best_model, _, _ = tune_lambda(evaluator)
    return best_model, best_lambda


def tune_posthoc(base_model, X_val, y_val):
    """Tune terminal correction while keeping the CART structure fixed."""
    means = np.array([leaf.value for leaf in base_model.leaves_], dtype=float)
    counts = np.array([leaf.n_samples for leaf in base_model.leaves_], dtype=float)
    val_ids = base_model.apply(X_val)

    def evaluator(lam):
        values = soft_calibrate(means, counts, base_model.edge_weights_, lam)
        return rmse(y_val, values[val_ids]), values

    best_lambda, best_values, _, _ = tune_lambda(evaluator)
    return best_lambda, best_values


def run_repetition(scenario, repetition, seed, confidence_method="entropy", n_experts=100):
    rng = np.random.default_rng(seed)
    X_dev, y_dev = generate_tree_regression(
        480, rng, local_violation=scenario.local_violation
    )
    X_test, y_test = generate_tree_regression(
        3000, rng, local_violation=scenario.local_violation
    )

    order = rng.permutation(len(y_dev))
    train_idx = order[:340]
    val_idx = order[340:]
    center, scale = response_standardization(y_dev[train_idx])
    y_train = (y_dev[train_idx] - center) / scale
    y_val = (y_dev[val_idx] - center) / scale

    panel_center = (
        TRUE_DIRECTIONS if scenario.panel_center == "truth" else -TRUE_DIRECTIONS
    )
    panel = simulate_expert_panel(
        panel_center,
        n_experts,
        scenario.error_rate,
        scenario.abstention_rate,
        rng,
    )
    directions, confidences, _ = aggregate_expert_votes(
        panel, method=confidence_method
    )
    informed = TRUE_DIRECTIONS != 0
    shared = {
        "scenario": scenario.name,
        "repetition": repetition,
        "mean_confidence": float(np.mean(confidences[informed])),
        "direction_accuracy": float(
            np.mean(directions[informed] == TRUE_DIRECTIONS[informed])
        ),
    }

    rows = []
    base = make_model().fit(X_dev[train_idx], y_train, directions, confidences)
    cart_pred = base.predict(X_test) * scale + center
    rows.append(
        {
            **shared,
            "variant": "cart",
            "rmse": rmse(y_test, cart_pred),
            "selected_lambda": 0.0,
            "n_leaves": len(base.leaves_),
        }
    )

    post_lambda, post_values = tune_posthoc(base, X_dev[val_idx], y_val)
    post_pred = post_values[base.apply(X_test)] * scale + center
    rows.append(
        {
            **shared,
            "variant": "posthoc",
            "rmse": rmse(y_test, post_pred),
            "selected_lambda": post_lambda,
            "n_leaves": len(base.leaves_),
        }
    )

    full, full_lambda = fit_full(
        X_dev[train_idx],
        y_train,
        X_dev[val_idx],
        y_val,
        directions,
        confidences,
    )
    full_pred = full.predict(X_test) * scale + center
    rows.append(
        {
            **shared,
            "variant": "full",
            "rmse": rmse(y_test, full_pred),
            "selected_lambda": full_lambda,
            "n_leaves": len(full.leaves_),
        }
    )

    hard = make_model(hard=True).fit(
        X_dev[train_idx], y_train, directions, confidences
    )
    hard_pred = hard.predict(X_test) * scale + center
    rows.append(
        {
            **shared,
            "variant": "hard",
            "rmse": rmse(y_test, hard_pred),
            "selected_lambda": np.nan,
            "n_leaves": len(hard.leaves_),
        }
    )
    return rows


def paired_summary(df):
    rows = []
    for scenario in [s.name for s in SCENARIOS]:
        block = df[df["scenario"] == scenario]
        pivot = block.pivot(index="repetition", columns="variant", values="rmse")
        for variant in ["cart", "posthoc", "full", "hard"]:
            values = pivot[variant].to_numpy()
            if variant == "cart":
                diff_mean = 0.0
                low = high = np.nan
            else:
                diff = (pivot[variant] - pivot["cart"]).to_numpy()
                diff_mean = float(np.mean(diff))
                se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
                crit = float(t.ppf(0.975, len(diff) - 1))
                low, high = diff_mean - crit * se, diff_mean + crit * se
            rows.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "mean_rmse": float(np.mean(values)),
                    "difference_vs_cart": diff_mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def plot_differences(summary, figures):
    scenario_labels = [
        "Aligned",
        "Local violation",
        "Uncertain panel",
        "Confidently wrong",
    ]
    scenario_names = [s.name for s in SCENARIOS]

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(len(scenario_names))
    for offset, variant in [(-0.10, "posthoc"), (0.10, "full")]:
        block = (
            summary[summary["variant"] == variant]
            .set_index("scenario")
            .loc[scenario_names]
        )
        y = block["difference_vs_cart"].to_numpy()
        err = np.vstack(
            [
                y - block["ci95_low"].to_numpy(),
                block["ci95_high"].to_numpy() - y,
            ]
        )
        ax.errorbar(
            x + offset,
            y,
            yerr=err,
            marker="o",
            capsize=3,
            linestyle="none",
            label="Post-hoc" if variant == "posthoc" else "Full",
        )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x, scenario_labels, rotation=15)
    ax.set_ylabel("RMSE difference vs CART")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "larger_tree_soft.png", dpi=220)
    fig.savefig(figures / "larger_tree_soft.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    block = (
        summary[summary["variant"] == "hard"]
        .set_index("scenario")
        .loc[scenario_names]
    )
    y = block["difference_vs_cart"].to_numpy()
    err = np.vstack(
        [
            y - block["ci95_low"].to_numpy(),
            block["ci95_high"].to_numpy() - y,
        ]
    )
    ax.errorbar(x, y, yerr=err, marker="o", capsize=3, linestyle="none")
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x, scenario_labels, rotation=15)
    ax.set_ylabel("Hard-order RMSE difference vs CART")
    fig.tight_layout()
    fig.savefig(figures / "larger_tree_hard.png", dpi=220)
    fig.savefig(figures / "larger_tree_hard.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    master = np.random.default_rng(args.seed)
    rows = []
    for scenario in SCENARIOS:
        for repetition in range(args.repetitions):
            seed = int(master.integers(0, 2**31 - 1))
            rows.extend(run_repetition(scenario, repetition, seed))

    df = pd.DataFrame(rows)
    summary = paired_summary(df)
    df.to_csv(args.results / "larger_tree_repetitions.csv", index=False)
    summary.to_csv(args.results / "larger_tree_summary.csv", index=False)
    plot_differences(summary, args.figures)


if __name__ == "__main__":
    main()
