"""Helpers shared by the numerical experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


# The response is standardized before tuning.  The base grid is deliberately
# broad; if its upper endpoint is selected, tune_lambda expands it by factors
# of ten until the validation optimum is no longer on the boundary (or the
# safety cap is reached).
LAMBDA_GRID = (0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)
MAX_LAMBDA = 10_000_000_000.0


def simulate_expert_panel(true_directions, n_experts, error_rate, abstention_rate, rng):
    """Generate independent {-1, 0, +1} votes around reference directions."""
    true_directions = np.asarray(true_directions, dtype=int)
    if n_experts <= 0:
        raise ValueError("n_experts must be positive")
    if not 0.0 <= error_rate <= 1.0 or not 0.0 <= abstention_rate <= 1.0:
        raise ValueError("error_rate and abstention_rate must lie in [0, 1]")
    if not np.isin(true_directions, (-1, 0, 1)).all():
        raise ValueError("true_directions must contain only -1, 0, +1")

    votes = np.zeros((n_experts, len(true_directions)), dtype=int)
    for e in range(n_experts):
        for j, direction in enumerate(true_directions):
            if direction == 0 or rng.random() < abstention_rate:
                continue
            votes[e, j] = -direction if rng.random() < error_rate else direction
    return votes


def generate_tree_regression(n, rng, noise_sd=1.35, local_violation=0.0):
    """Six-predictor regression model with an optional local reversal in X1."""
    X = rng.uniform(0.0, 1.0, size=(n, 6))
    signal = (
        2.8 * (X[:, 0] > 0.48)
        + 1.8 * (X[:, 1] > 0.32)
        + 1.3 * X[:, 2]
        - 1.4 * X[:, 3]
        + 0.4 * X[:, 0] * X[:, 1]
        + local_violation * np.sin(4.0 * np.pi * X[:, 0])
    )
    y = signal + rng.normal(0.0, noise_sd, size=n)
    return X, y


def response_standardization(y_train):
    """Return the training-sample center and scale used before lambda tuning."""
    y_train = np.asarray(y_train, dtype=float)
    center = float(np.mean(y_train))
    scale = float(np.std(y_train))
    return center, scale if scale > 0.0 else 1.0


def rmse(y_true, y_pred):
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def tune_lambda(
    evaluator: Callable[[float], tuple[float, Any]],
    base_grid=LAMBDA_GRID,
    max_lambda: float = MAX_LAMBDA,
    improvement_tol: float = 1e-10,
):
    """Minimize validation loss over a grid, expanding if the optimum is on its upper edge.

    ``evaluator(lambda_)`` must return ``(score, payload)``.  The smallest
    lambda is retained when scores agree within ``improvement_tol``.
    """
    grid = [float(x) for x in base_grid]
    if not grid or grid[0] != 0.0 or any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("base_grid must be strictly increasing and start at zero")

    best_score = np.inf
    best_lambda = None
    best_payload = None
    evaluated = []

    def evaluate_one(lam: float):
        nonlocal best_score, best_lambda, best_payload
        score, payload = evaluator(lam)
        score = float(score)
        if not np.isfinite(score):
            raise RuntimeError(f"non-finite validation score for lambda={lam}")
        evaluated.append((lam, score))
        if score < best_score - improvement_tol:
            best_score = score
            best_lambda = lam
            best_payload = payload

    for lam in grid:
        evaluate_one(lam)

    # Boundary expansion is rare but important for low-confidence panels,
    # because the effective penalty is lambda * q_j.
    upper = grid[-1]
    while best_lambda == upper and upper < max_lambda:
        next_upper = min(10.0 * upper, max_lambda)
        if next_upper <= upper:
            break
        evaluate_one(next_upper)
        upper = next_upper

    if best_lambda == upper and upper >= max_lambda:
        raise RuntimeError(
            "validation optimum remains at the maximum lambda; increase MAX_LAMBDA"
        )

    return best_lambda, best_payload, best_score, tuple(evaluated)
