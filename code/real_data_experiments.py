"""Public-data case studies and deletion-stability diagnostics.

This version evaluates eight regression datasets:
    1. Diabetes
    2. Engel food expenditure
    3. Concrete compressive strength
    4. Abalone
    5. Red wine quality
    6. White wine quality
    7. Auto MPG
    8. Windsor housing

The additional datasets are chosen to provide several distinct regression
settings with interpretable directional information.  Where possible, the
reference directions follow monotonic features used in earlier literature;
they are still treated here as uncertain qualitative information rather than
as hard scientific laws.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t
from sklearn.datasets import fetch_openml, load_diabetes

from simulation_utils import (
    response_standardization,
    rmse,
    simulate_expert_panel,
    tune_lambda,
)
from stability_diagnostics import deletion_stability
from uetree import UncertainExpertRegressionTree, aggregate_expert_votes, soft_calibrate
from dataset_snapshots import load_processed_datasets

@dataclass(frozen=True)
class DatasetSpec:
    name: str
    X: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    directions: np.ndarray
    min_leaf: int


DISPLAY_NAMES = {
    "diabetes": "Diabetes",
    "engel": "Engel",
    "concrete": "Concrete",
    "abalone": "Abalone",
    "red_wine": "Red wine",
    "white_wine": "White wine",
    "auto_mpg": "Auto MPG",
    "windsor": "Windsor",
}


def _normalize_name(name: str) -> str:
    """Normalize feature names so priors are robust to spaces/underscores/case."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _prepare_spec(
    name,
    X,
    y,
    prior,
    min_leaf,
    *,
    drop_columns=(),
):
    """Convert a pandas dataset to the numeric format used by the tree code."""
    X = pd.DataFrame(X).copy()

    drop_norm = {_normalize_name(col) for col in drop_columns}
    to_drop = [col for col in X.columns if _normalize_name(col) in drop_norm]
    if to_drop:
        X = X.drop(columns=to_drop)

    # Keep all available predictors. Categorical variables are represented by
    # one-hot indicators and receive zero expert weight unless explicitly named.
    X = pd.get_dummies(X, drop_first=False, dtype=float)
    feature_names = tuple(str(col) for col in X.columns)

    if isinstance(y, pd.DataFrame):
        raise ValueError(
            f"Dataset {name!r} still has multiple target columns. "
            "Select one target before calling _prepare_spec()."
        )
    y = pd.to_numeric(pd.Series(y), errors="coerce")

    X_array = X.to_numpy(dtype=float)
    y_array = y.to_numpy(dtype=float)
    if len(y_array) != len(X_array):
        raise ValueError(
            f"Dataset {name!r} has incompatible X/y lengths: "
            f"{len(X_array)} predictors rows and {len(y_array)} target values."
        )

    finite = np.isfinite(y_array) & np.all(np.isfinite(X_array), axis=1)
    X_array = X_array[finite]
    y_array = y_array[finite]
    if len(y_array) == 0:
        raise ValueError(
            f"Dataset {name!r} contains no usable observations after numeric "
            "conversion and removal of non-finite values."
        )

    prior_norm = {_normalize_name(key): int(value) for key, value in prior.items()}
    directions = np.array(
        [prior_norm.get(_normalize_name(col), 0) for col in feature_names], dtype=int
    )

    if not np.any(directions != 0):
        raise ValueError(
            f"No directional prior matched the features of dataset {name!r}. "
            f"Available features: {feature_names}"
        )

    return DatasetSpec(
        name=name,
        X=X_array,
        y=y_array,
        feature_names=feature_names,
        directions=directions,
        min_leaf=min_leaf,
    )


def _select_target(target, target_name=None):
    """Select one numeric target from an OpenML Series/DataFrame target object."""
    if isinstance(target, pd.DataFrame):
        if target_name is None:
            if target.shape[1] != 1:
                raise ValueError(
                    "OpenML returned multiple targets; target_name must be specified."
                )
            return target.iloc[:, 0]

        wanted = _normalize_name(target_name)
        matches = [col for col in target.columns if _normalize_name(col) == wanted]
        if not matches:
            raise ValueError(
                f"Target {target_name!r} not found. Available targets: "
                f"{tuple(str(c) for c in target.columns)}"
            )
        return target[matches[0]]

    return target


def _load_openml_spec(
    data_id,
    name,
    prior,
    min_leaf,
    data_home,
    *,
    target_name=None,
    drop_columns=(),
):
    # If an explicit target name is supplied, fetch all OpenML columns first
    # and select the target ourselves. This is more robust for datasets whose
    # OpenML metadata do not define a default target (e.g. wine ID 42184).
    if target_name is not None:
        bunch = fetch_openml(
            data_id=data_id,
            target_column=None,
            as_frame=True,
            data_home=str(data_home),
        )
        frame = pd.DataFrame(bunch.data).copy()
        wanted = _normalize_name(target_name)
        matches = [col for col in frame.columns if _normalize_name(col) == wanted]
        if not matches:
            raise ValueError(
                f"Target {target_name!r} not found in OpenML dataset {data_id}. "
                f"Available columns: {tuple(str(c) for c in frame.columns)}"
            )
        target_col = matches[0]
        y = frame[target_col].copy()
        X = frame.drop(columns=[target_col])
    else:
        bunch = fetch_openml(
            data_id=data_id,
            as_frame=True,
            data_home=str(data_home),
        )
        y = bunch.target
        X = bunch.data
        if y is None or len(y) == 0:
            raise ValueError(
                f"OpenML dataset {data_id} ({name}) has no default target. "
                "Specify target_name explicitly in _load_openml_spec()."
            )

    return _prepare_spec(
        name,
        X,
        y,
        prior,
        min_leaf,
        drop_columns=drop_columns,
    )


def _load_cached_csv(url: str, cache_path: Path) -> pd.DataFrame:
    """Read a small public CSV once and reuse a local cached copy afterwards."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_csv(cache_path)

    frame = pd.read_csv(url)
    frame.to_csv(cache_path, index=False)
    return frame


def load_datasets(data_home: Path):
    """Load the eight public regression datasets used in the case studies."""
    data_home.mkdir(parents=True, exist_ok=True)

    # 1. Diabetes: same reference directions as in the original experiment.
    diabetes = load_diabetes(as_frame=True)
    diabetes_spec = _prepare_spec(
        "diabetes",
        diabetes.data,
        diabetes.target,
        {"bmi": 1, "bp": 1, "s5": 1, "s3": -1},
        12,
    )

    # 2. Engel food expenditure: higher household income -> higher food expenditure.
    engel = sm.datasets.engel.load_pandas().data
    engel_spec = _prepare_spec(
        "engel",
        engel[["income"]],
        engel["foodexp"],
        {"income": 1},
        10,
    )

    # 3. Concrete compressive strength, OpenML 44959.
    # Broad reference directions: more cement and greater age -> greater strength.
    concrete_spec = _load_openml_spec(
        44959,
        "concrete",
        {"cement": 1, "age": 1},
        15,
        data_home,
        target_name="strength",
    )

    # 4. Abalone, OpenML 183.
    # Monotonic features used in the literature: Length, Diameter, Height,
    # and Shell weight, all with positive direction for the ring count.
    abalone_spec = _load_openml_spec(
        183,
        "abalone",
        {
            "length": 1,
            "diameter": 1,
            "height": 1,
            "shell_weight": 1,
            "shell weight": 1,
        },
        20,
        data_home,
        target_name="Class_number_of_rings",
    )

    # 5. Red wine quality, OpenML 42184.
    # Directions follow the qualitative monotonic features used by Kokel et al.:
    # volatile acidity (-), citric acid (+), sulphates (+), alcohol (+).
    red_wine_spec = _load_openml_spec(
        42184,
        "red_wine",
        {
            "volatile acidity": -1,
            "volatileacidity": -1,
            "citric acid": 1,
            "citricacid": 1,
            "sulphates": 1,
            "alcohol": 1,
        },
        15,
        data_home,
        target_name="quality",
    )

    # 6. White wine quality, OpenML 40498.
    # This OpenML version uses generic feature names V1,...,V11 and target Class.
    # UCI order: V2=volatile acidity, V3=citric acid, V10=sulphates, V11=alcohol.
    white_wine_spec = _load_openml_spec(
        40498,
        "white_wine",
        {"v2": -1, "v3": 1, "v10": 1, "v11": 1},
        20,
        data_home,
        target_name="Class",
    )

    # 7. Auto MPG (ISLR/Rdatasets; 392 complete observations).
    # MPG is treated as decreasing with displacement, horsepower, and weight.
    auto = _load_cached_csv(
        "https://vincentarelbundock.github.io/Rdatasets/csv/ISLR/Auto.csv",
        data_home / "auto_mpg_islr.csv",
    )
    auto_mpg_spec = _prepare_spec(
        "auto_mpg",
        auto.drop(columns=["mpg"], errors="ignore"),
        auto["mpg"],
        {"displacement": -1, "horsepower": -1, "weight": -1},
        10,
        drop_columns=("rownames", "name"),
    )

    # 8. Windsor housing (Ecdat/Rdatasets; 546 observations).
    # Following the monotonic benchmark used in Kokel et al., lot size is the
    # informed predictor and is assigned a positive direction for sale price.
    windsor = _load_cached_csv(
        "https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Housing.csv",
        data_home / "windsor_housing_ecdat.csv",
    )
    windsor_spec = _prepare_spec(
        "windsor",
        windsor.drop(columns=["price"], errors="ignore"),
        windsor["price"],
        {"lotsize": 1, "lot size": 1},
        10,
        drop_columns=("rownames",),
    )

    return [
        diabetes_spec,
        engel_spec,
        concrete_spec,
        abalone_spec,
        red_wine_spec,
        white_wine_spec,
        auto_mpg_spec,
        windsor_spec,
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
                rel_mean, rel_low, rel_high = 0.0, np.nan, np.nan
            else:
                # Absolute paired RMSE difference, retained for the numerical table.
                diff = (pivot[variant] - pivot["cart"]).to_numpy()
                diff_mean = float(np.mean(diff))
                se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
                crit = float(t.ppf(0.975, len(diff) - 1))
                low, high = diff_mean - crit * se, diff_mean + crit * se

                # Relative paired change makes effects comparable across datasets
                # whose responses are measured on very different scales.
                rel = 100.0 * (pivot[variant] / pivot["cart"] - 1.0).to_numpy()
                rel_mean = float(np.mean(rel))
                rel_se = float(np.std(rel, ddof=1) / np.sqrt(len(rel)))
                rel_low = rel_mean - crit * rel_se
                rel_high = rel_mean + crit * rel_se

            rows.append(
                {
                    "dataset": dataset,
                    "panel": panel,
                    "variant": variant,
                    "mean_rmse": float(np.mean(values)),
                    "difference_vs_cart": diff_mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "relative_difference_pct": rel_mean,
                    "relative_ci95_low": rel_low,
                    "relative_ci95_high": rel_high,
                }
            )
    return pd.DataFrame(rows)

def summarize_confidence(df):
    return (
        df.groupby(["dataset", "panel"], sort=False)["mean_confidence"]
        .mean()
        .reset_index(name="mean_featurewise_confidence")
    )


def _extract_plot_values(summary, dataset_order, panel, variant):
    ys, lows, highs = [], [], []
    for dataset in dataset_order:
        row = summary[
            (summary["dataset"] == dataset)
            & (summary["panel"] == panel)
            & (summary["variant"] == variant)
        ].iloc[0]
        ys.append(row["difference_vs_cart"])
        lows.append(row["ci95_low"])
        highs.append(row["ci95_high"])
    ys = np.asarray(ys, dtype=float)
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    err = np.vstack([ys - lows, highs - ys])
    return ys, err


def _plot_one_panel(summary, figures, dataset_order, panel):
    """Create a readable panel-specific figure for all eight datasets."""
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(dataset_order))
    labels = [DISPLAY_NAMES.get(d, d.title()) for d in dataset_order]

    for offset, variant in [(-0.08, "posthoc"), (0.08, "full")]:
        ys, err = _extract_plot_values(summary, dataset_order, panel, variant)
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
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("RMSE difference vs CART")
    ax.set_title(f"{panel.capitalize()} expert panel")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / f"real_data_rmse_difference_{panel}.png", dpi=220)
    fig.savefig(figures / f"real_data_rmse_difference_{panel}.pdf")
    plt.close(fig)


def plot_results(summary, figures):
    """Plot all datasets and additionally save reliable/uncertain figures separately."""
    dataset_order = list(dict.fromkeys(summary["dataset"].tolist()))

    # Main figure: one x-position per dataset and four soft-method/panel combinations.
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    x = np.arange(len(dataset_order))
    labels = [DISPLAY_NAMES.get(d, d.title()) for d in dataset_order]
    series = [
        (-0.18, "reliable", "posthoc", "Reliable: Post-hoc"),
        (-0.06, "reliable", "full", "Reliable: Full"),
        (0.06, "uncertain", "posthoc", "Uncertain: Post-hoc"),
        (0.18, "uncertain", "full", "Uncertain: Full"),
    ]

    for offset, panel, variant, label in series:
        ys, err = _extract_plot_values(summary, dataset_order, panel, variant)
        ax.errorbar(
            x + offset,
            ys,
            yerr=err,
            marker="o",
            capsize=3,
            linestyle="none",
            label=label,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("RMSE difference vs CART")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "real_data_rmse_difference.png", dpi=220)
    fig.savefig(figures / "real_data_rmse_difference.pdf")
    plt.close(fig)

    # Extra files make it easy to use two separate panels/subfigures in the paper.
    for panel in ["reliable", "uncertain"]:
        _plot_one_panel(summary, figures, dataset_order, panel)



def _extract_relative_plot_values(summary, dataset_order, panel, variant):
    ys, lows, highs = [], [], []
    for dataset in dataset_order:
        row = summary[
            (summary["dataset"] == dataset)
            & (summary["panel"] == panel)
            & (summary["variant"] == variant)
        ].iloc[0]
        ys.append(row["relative_difference_pct"])
        lows.append(row["relative_ci95_low"])
        highs.append(row["relative_ci95_high"])
    ys = np.asarray(ys, dtype=float)
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)
    err = np.vstack([ys - lows, highs - ys])
    return ys, err


def _plot_one_panel_relative(summary, figures, dataset_order, panel):
    """Panel-specific relative RMSE change; directly comparable across datasets."""
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(dataset_order))
    labels = [DISPLAY_NAMES.get(d, d.title()) for d in dataset_order]

    for offset, variant in [(-0.08, "posthoc"), (0.08, "full")]:
        ys, err = _extract_relative_plot_values(summary, dataset_order, panel, variant)
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
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Relative RMSE change vs CART (%)")
    ax.set_title(f"{panel.capitalize()} expert panel")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figures / f"real_data_relative_rmse_difference_{panel}.png", dpi=220)
    fig.savefig(figures / f"real_data_relative_rmse_difference_{panel}.pdf")
    plt.close(fig)


def plot_relative_results(summary, figures):
    """Save scale-free figures based on paired percentage RMSE changes."""
    dataset_order = list(dict.fromkeys(summary["dataset"].tolist()))

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    x = np.arange(len(dataset_order))
    labels = [DISPLAY_NAMES.get(d, d.title()) for d in dataset_order]
    series = [
        (-0.18, "reliable", "posthoc", "Reliable: Post-hoc"),
        (-0.06, "reliable", "full", "Reliable: Full"),
        (0.06, "uncertain", "posthoc", "Uncertain: Post-hoc"),
        (0.18, "uncertain", "full", "Uncertain: Full"),
    ]

    for offset, panel, variant, label in series:
        ys, err = _extract_relative_plot_values(summary, dataset_order, panel, variant)
        ax.errorbar(
            x + offset,
            ys,
            yerr=err,
            marker="o",
            capsize=3,
            linestyle="none",
            label=label,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Relative RMSE change vs CART (%)")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "real_data_relative_rmse_difference.png", dpi=220)
    fig.savefig(figures / "real_data_relative_rmse_difference.pdf")
    plt.close(fig)

    for panel in ["reliable", "uncertain"]:
        _plot_one_panel_relative(summary, figures, dataset_order, panel)

def _latex_escape(text):
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def write_latex_table(summary, path):
    """Write a ready-to-paste LaTeX tabular with all eight datasets."""
    method_names = {"cart": "CART", "posthoc": "Post-hoc", "full": "Full"}
    lines = [
        r"\begin{tabular}{lllrrl}",
        r"\toprule",
        r"Dataset & Panel & Method & Mean RMSE & Difference & 95\% CI \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        dataset = _latex_escape(DISPLAY_NAMES.get(row["dataset"], row["dataset"]))
        panel = _latex_escape(row["panel"])
        method = method_names[row["variant"]]
        mean_rmse = f"{row['mean_rmse']:.3f}"

        if row["variant"] == "cart":
            difference = "0"
            ci = r"--"
        else:
            difference = f"{row['difference_vs_cart']:.3f}"
            ci = f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"

        lines.append(
            f"{dataset} & {panel} & {method} & {mean_rmse} & {difference} & {ci} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_relative_latex_table(summary, path):
    """Write a compact LaTeX table with scale-free percentage changes vs CART."""
    method_names = {"cart": "CART", "posthoc": "Post-hoc", "full": "Full"}
    lines = [
        r"\begin{tabular}{lllrl}",
        r"\toprule",
        r"Dataset & Panel & Method & Relative difference (\%) & 95\% CI \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        dataset = _latex_escape(DISPLAY_NAMES.get(row["dataset"], row["dataset"]))
        panel = _latex_escape(row["panel"])
        method = method_names[row["variant"]]

        if row["variant"] == "cart":
            difference = "0"
            ci = r"--"
        else:
            difference = f"{row['relative_difference_pct']:.2f}"
            ci = f"[{row['relative_ci95_low']:.2f}, {row['relative_ci95_high']:.2f}]"

        lines.append(f"{dataset} & {panel} & {method} & {difference} & {ci} " + r"\\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    parser.add_argument("--data-home", type=Path, default=Path("data_cache"))
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--dataset-snapshot", type=Path, default=None)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)


    specs = (
        load_processed_datasets(args.dataset_snapshot)
        if args.dataset_snapshot is not None
        else load_datasets(args.data_home)
    )
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
    confidence_summary = summarize_confidence(df)

    df.to_csv(args.results / "real_data_repetitions.csv", index=False)
    summary.to_csv(args.results / "real_data_summary.csv", index=False)
    confidence_summary.to_csv(
        args.results / "real_data_confidence_summary.csv", index=False
    )
    write_latex_table(summary, args.results / "real_data_summary.tex")
    write_relative_latex_table(
        summary, args.results / "real_data_relative_summary.tex"
    )
    plot_results(summary, args.figures)
    plot_relative_results(summary, args.figures)
    run_stability(specs, args.results, args.seed + 1000)


if __name__ == "__main__":
    main()
