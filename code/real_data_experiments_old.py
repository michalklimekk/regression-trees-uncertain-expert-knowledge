"""Public-data case studies and deletion-stability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from sklearn.datasets import load_diabetes

from simulation_utils import (
    response_standardization,
    rmse,
    simulate_expert_panel,
    tune_lambda,
)
from stability_diagnostics import deletion_stability
from uetree import UncertainExpertRegressionTree, aggregate_expert_votes, soft_calibrate


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    directions: np.ndarray
    min_leaf: int


def load_datasets():
    diabetes = load_diabetes()
    names = tuple(str(x) for x in diabetes.feature_names)
    prior = {"bmi": 1, "bp": 1, "s5": 1, "s3": -1}
    diabetes_directions = np.array(
        [prior.get(name, 0) for name in names], dtype=int
    )

    engel = sm.datasets.engel.load_pandas().data
    return [
        DatasetSpec(
            "diabetes",
            np.asarray(diabetes.data, dtype=float),
            np.asarray(diabetes.target, dtype=float),
            names,
            diabetes_directions,
            12,
        ),
        DatasetSpec(
            "engel",
            engel[["income"]].to_numpy(dtype=float),
            engel["foodexp"].to_numpy(dtype=float),
            ("income",),
            np.array([1], dtype=int),
            10,
        ),
    ]


def make_model(min_leaf, split_lambda=0.0, leaf_lambda=0.0):
    return UncertainExpertRegressionTree(
        max_depth=4,
        min_samples_leaf=min_leaf,
        max_thresholds=40,
        split_lambda=split_lambda,
        leaf_lambda=leaf_lambda,
    )


def fit_full(X_train, y_train, X_val, y_val, directions, confidences, min_leaf):
    def evaluator(lam):
        model = make_model(min_leaf, lam, lam).fit(
            X_train, y_train, directions, confidences
        )
        return rmse(y_val, model.predict(X_val)), model

    best_lambda, best_model, _, _ = tune_lambda(evaluator)
    return best_model, best_lambda


def tune_posthoc(base_model, X_val, y_val):
    means = np.array([leaf.value for leaf in base_model.leaves_], dtype=float)
    counts = np.array([leaf.n_samples for leaf in base_model.leaves_], dtype=float)
    val_ids = base_model.apply(X_val)

    def evaluator(lam):
        values = soft_calibrate(means, counts, base_model.edge_weights_, lam)
        return rmse(y_val, values[val_ids]), values

    best_lambda, best_values, _, _ = tune_lambda(evaluator)
    return best_lambda, best_values


def run_case(spec, panel_name, error_rate, abstention_rate, repetition, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(spec.y))
    n = len(order)
    n_test = int(round(0.25 * n))
    n_val = int(round(0.1875 * n))
    test_idx = order[:n_test]
    val_idx = order[n_test : n_test + n_val]
    train_idx = order[n_test + n_val :]

    center, scale = response_standardization(spec.y[train_idx])
    y_train = (spec.y[train_idx] - center) / scale
    y_val = (spec.y[val_idx] - center) / scale

    panel = simulate_expert_panel(
        spec.directions, 100, error_rate, abstention_rate, rng
    )
    directions, confidences, _ = aggregate_expert_votes(panel)
    informed = spec.directions != 0
    shared = {
        "dataset": spec.name,
        "panel": panel_name,
        "repetition": repetition,
        "mean_confidence": float(np.mean(confidences[informed])),
    }

    rows = []
    base = make_model(spec.min_leaf).fit(
        spec.X[train_idx], y_train, directions, confidences
    )
    cart_pred = base.predict(spec.X[test_idx]) * scale + center
    rows.append(
        {
            **shared,
            "variant": "cart",
            "rmse": rmse(spec.y[test_idx], cart_pred),
            "selected_lambda": 0.0,
        }
    )

    post_lambda, post_values = tune_posthoc(base, spec.X[val_idx], y_val)
    post_pred = post_values[base.apply(spec.X[test_idx])] * scale + center
    rows.append(
        {
            **shared,
            "variant": "posthoc",
            "rmse": rmse(spec.y[test_idx], post_pred),
            "selected_lambda": post_lambda,
        }
    )

    full, full_lambda = fit_full(
        spec.X[train_idx],
        y_train,
        spec.X[val_idx],
        y_val,
        directions,
        confidences,
        spec.min_leaf,
    )
    full_pred = full.predict(spec.X[test_idx]) * scale + center
    rows.append(
        {
            **shared,
            "variant": "full",
            "rmse": rmse(spec.y[test_idx], full_pred),
            "selected_lambda": full_lambda,
        }
    )
    return rows


def summarize(df):
    rows = []
    for (dataset, panel), block in df.groupby(["dataset", "panel"], sort=False):
        pivot = block.pivot(index="repetition", columns="variant", values="rmse")
        for variant in ["cart", "posthoc", "full"]:
            values = pivot[variant].to_numpy()
            if variant == "cart":
                diff_mean, low, high = 0.0, np.nan, np.nan
            else:
                diff = (pivot[variant] - pivot["cart"]).to_numpy()
                diff_mean = float(np.mean(diff))
                se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
                crit = float(t.ppf(0.975, len(diff) - 1))
                low, high = diff_mean - crit * se, diff_mean + crit * se
            rows.append(
                {
                    "dataset": dataset,
                    "panel": panel,
                    "variant": variant,
                    "mean_rmse": float(np.mean(values)),
                    "difference_vs_cart": diff_mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def plot_results(summary, figures):
    datasets = ["diabetes", "engel"]
    panels = ["reliable", "uncertain"]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(4)
    labels = [
        "Diabetes\nreliable",
        "Diabetes\nuncertain",
        "Engel\nreliable",
        "Engel\nuncertain",
    ]
    combinations = [(d, p) for d in datasets for p in panels]
    for offset, variant in [(-0.08, "posthoc"), (0.08, "full")]:
        ys, lows, highs = [], [], []
        for dataset, panel in combinations:
            row = summary[
                (summary["dataset"] == dataset)
                & (summary["panel"] == panel)
                & (summary["variant"] == variant)
            ].iloc[0]
            ys.append(row["difference_vs_cart"])
            lows.append(row["ci95_low"])
            highs.append(row["ci95_high"])
        ys = np.asarray(ys, dtype=float)
        err = np.vstack([ys - np.asarray(lows), np.asarray(highs) - ys])
        ax.errorbar(
            x + offset,
            ys,
            yerr=err,
            marker="o",
            capsize=3,
            linestyle="none",
            label="Post-hoc" if variant == "posthoc" else "Full",
        )
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE difference vs CART")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / "real_data_rmse_difference.png", dpi=220)
    fig.savefig(figures / "real_data_rmse_difference.pdf")
    plt.close(fig)


def run_stability(specs, results, seed):
    rows = []
    for i, spec in enumerate(specs):
        directions, stability = deletion_stability(
            spec.X,
            spec.y,
            n_resamples=500,
            remove_fraction=0.10,
            min_samples_leaf=spec.min_leaf,
            seed=seed + i,
        )
        for j, name in enumerate(spec.feature_names):
            if spec.directions[j] == 0:
                continue
            rows.append(
                {
                    "dataset": spec.name,
                    "predictor": name,
                    "reference_direction": int(spec.directions[j]),
                    "modal_stump_direction": int(directions[j]),
                    "stability": float(stability[j]),
                }
            )
    pd.DataFrame(rows).to_csv(results / "deletion_stability.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    specs = load_datasets()
    panel_settings = [
        ("reliable", 0.10, 0.05),
        ("uncertain", 0.30, 0.20),
    ]
    master = np.random.default_rng(args.seed)
    rows = []
    for spec in specs:
        seeds = [
            int(master.integers(0, 2**31 - 1)) for _ in range(args.repetitions)
        ]
        for panel_name, error_rate, abstention_rate in panel_settings:
            for repetition, seed in enumerate(seeds):
                rows.extend(
                    run_case(
                        spec,
                        panel_name,
                        error_rate,
                        abstention_rate,
                        repetition,
                        seed,
                    )
                )

    df = pd.DataFrame(rows)
    summary = summarize(df)
    df.to_csv(args.results / "real_data_repetitions.csv", index=False)
    summary.to_csv(args.results / "real_data_summary.csv", index=False)
    plot_results(summary, args.figures)
    run_stability(specs, args.results, args.seed + 1000)


if __name__ == "__main__":
    main()
