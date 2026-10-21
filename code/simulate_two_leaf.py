"""Numerical verification of the exact two-leaf Gaussian risk formula."""

from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def soft_order(d, tau):
    """One-sided threshold map S_tau applied elementwise."""
    d = np.asarray(d, dtype=float)
    return np.where(d <= 0.0, d, np.where(d <= tau, 0.0, d - tau))


def exact_risk(delta, sigma, n_a, n_b, tau):
    """Exact weighted squared-error risk from the two-leaf theorem."""
    delta = np.asarray(delta, dtype=float)
    h = n_a * n_b / (n_a + n_b)
    s = sigma / np.sqrt(h)
    z0 = -delta / s
    z1 = (tau - delta) / s
    part = s**2 * (norm.cdf(z0) - z0 * norm.pdf(z0))
    part += delta**2 * (norm.cdf(z1) - norm.cdf(z0))
    part += s**2 * (1.0 - norm.cdf(z1) + z1 * norm.pdf(z1))
    part += -2.0 * s * tau * norm.pdf(z1)
    part += tau**2 * (1.0 - norm.cdf(z1))
    return sigma**2 + h * part


def risk_crossing(sigma, n_a, n_b, tau):
    """Positive Delta where the regularized and unregularized risks cross."""
    baseline = 2.0 * sigma**2

    def difference(x):
        return float(exact_risk(x, sigma, n_a, n_b, tau) - baseline)

    hi = sigma
    while difference(hi) < 0.0:
        hi *= 2.0
    return brentq(difference, 0.0, hi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--mc", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    sigma = 1.0
    n_a = n_b = 30
    h = n_a * n_b / (n_a + n_b)
    s = sigma / np.sqrt(h)
    delta_std = np.linspace(-3.0, 3.0, 241)
    deltas = delta_std * s
    ratios = [0.0, 0.5, 1.0, 1.5]

    rng = np.random.default_rng(args.seed)
    z = rng.normal(size=args.mc)
    rows = []
    summary = []

    for ratio in ratios:
        tau = ratio * s
        exact = exact_risk(deltas, sigma, n_a, n_b, tau)
        mc_values = []
        for delta in deltas:
            D = delta + s * z
            mc = sigma**2 + h * np.mean((soft_order(D, tau) - delta) ** 2)
            mc_values.append(mc)
        for ds, delta, er, mr in zip(delta_std, deltas, exact, mc_values):
            rows.append(
                {
                    "delta_over_s": ds,
                    "tau_over_s": ratio,
                    "exact_risk": float(er),
                    "mc_risk": float(mr),
                    "risk_ratio": float(er / (2.0 * sigma**2)),
                }
            )
        if ratio > 0:
            crossing = risk_crossing(sigma, n_a, n_b, tau)
            summary.append(
                {
                    "tau_over_s": ratio,
                    "risk_ratio_at_zero": float(
                        exact_risk(0.0, sigma, n_a, n_b, tau)
                        / (2.0 * sigma**2)
                    ),
                    "crossing_delta_over_s": float(crossing / s),
                    "max_abs_mc_error": float(
                        np.max(np.abs(exact - np.asarray(mc_values)))
                    ),
                }
            )

    df = pd.DataFrame(rows)
    pd.DataFrame(summary).to_csv(args.results / "two_leaf_summary.csv", index=False)
    df.to_csv(args.results / "two_leaf_risk.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for ratio in ratios:
        block = df[df["tau_over_s"] == ratio]
        ax.plot(
            block["delta_over_s"],
            block["risk_ratio"],
            label=rf"$\tau/s={ratio:.1f}$",
        )
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.axvline(0.0, linestyle=":", linewidth=1)
    ax.set_xlabel(r"true contrast $\Delta/s$")
    ax.set_ylabel(r"risk ratio $R_\gamma/R_0$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.figures / "two_leaf_risk.png", dpi=220)
    fig.savefig(args.figures / "two_leaf_risk.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
