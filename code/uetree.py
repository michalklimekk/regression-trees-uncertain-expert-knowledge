"""Regression tree with uncertain directional expert knowledge.

The implementation follows the manuscript:
- candidate splits use CART gain minus the exact two-child penalty increment;
- terminal values can be corrected by a confidence-weighted soft order penalty;
- a hard-order comparator is available for the controlled simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import LinearConstraint, minimize


EPS = 1e-12


@dataclass
class Node:
    indices: np.ndarray
    value: float
    n_samples: int
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    leaf_id: Optional[int] = None
    calibrated_value: Optional[float] = None

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def binary_entropy(p: float) -> float:
    """Binary entropy with the convention 0 log(0) = 0."""
    if p <= EPS or p >= 1.0 - EPS:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def aggregate_expert_votes(votes: np.ndarray, method: str = "entropy"):
    """Aggregate {-1, 0, +1} expert votes into directions and confidence weights."""
    votes = np.asarray(votes, dtype=int)
    if votes.ndim != 2 or not np.isin(votes, (-1, 0, 1)).all():
        raise ValueError("votes must be a two-dimensional array with entries -1, 0, +1")
    if votes.shape[0] == 0:
        raise ValueError("at least one expert is required")
    if method not in {"entropy", "margin", "coverage"}:
        raise ValueError("unknown confidence mapping")

    m, p = votes.shape
    directions = np.zeros(p, dtype=int)
    confidences = np.zeros(p, dtype=float)
    positive_share = np.full(p, np.nan, dtype=float)

    for j in range(p):
        active = votes[:, j] != 0
        n_active = int(active.sum())
        if n_active == 0:
            continue

        p_plus = float(np.mean(votes[active, j] == 1))
        positive_share[j] = p_plus
        if p_plus > 0.5:
            directions[j] = 1
        elif p_plus < 0.5:
            directions[j] = -1
        else:
            continue

        coverage = n_active / m
        if method == "entropy":
            concentration = 1.0 - binary_entropy(p_plus)
        elif method == "margin":
            concentration = abs(2.0 * p_plus - 1.0)
        else:
            concentration = 1.0
        confidences[j] = coverage * concentration

    return directions, confidences, positive_share


def penalty_increment(d: float, h: float, gamma: float) -> float:
    """Exact local objective increment A_gamma(d) from the two-leaf calculation."""
    if h <= 0.0:
        raise ValueError("h must be positive")
    if gamma <= EPS or d <= 0.0:
        return 0.0
    cutoff = gamma / h
    if d <= cutoff:
        return 0.5 * h * d * d
    return gamma * d - gamma * gamma / (2.0 * h)


def soft_calibrate(means, counts, edge_weights, penalty, tol=1e-10, max_iter=10000):
    """Solve the soft terminal-leaf calibration problem by dual coordinate descent."""
    means = np.asarray(means, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if means.ndim != 1 or counts.ndim != 1 or len(means) != len(counts):
        raise ValueError("means and counts must be one-dimensional arrays of equal length")
    if np.any(counts <= 0.0):
        raise ValueError("all leaf counts must be positive")
    if not np.isfinite(means).all() or not np.isfinite(counts).all():
        raise ValueError("means and counts must be finite")
    if penalty < 0.0:
        raise ValueError("penalty must be nonnegative")

    edges = [(a, b, float(w)) for (a, b), w in edge_weights.items() if w > 0.0]
    if penalty <= EPS or not edges:
        return means.copy()

    k = len(means)
    m = len(edges)
    B = np.zeros((m, k), dtype=float)
    upper = np.empty(m, dtype=float)
    for r, (a, b, weight) in enumerate(edges):
        if not (0 <= a < k and 0 <= b < k and a != b):
            raise ValueError("invalid leaf-relation edge")
        B[r, a] = 1.0
        B[r, b] = -1.0
        upper[r] = penalty * weight

    inv_counts = 1.0 / counts
    M = (B * inv_counts[None, :]) @ B.T
    linear = B @ means
    alpha = np.zeros(m, dtype=float)
    gradient = -linear.copy()
    diagonal = np.diag(M)

    converged = False
    for _ in range(max_iter):
        max_change = 0.0
        for r in range(m):
            curvature = diagonal[r]
            if curvature <= EPS or upper[r] <= EPS:
                continue
            new_value = np.clip(alpha[r] - gradient[r] / curvature, 0.0, upper[r])
            change = float(new_value - alpha[r])
            if change != 0.0:
                alpha[r] = new_value
                gradient += change * M[:, r]
                max_change = max(max_change, abs(change))
        if max_change < tol:
            converged = True
            break

    if not converged:
        raise RuntimeError("soft leaf calibration did not converge")

    return means - inv_counts * (B.T @ alpha)


def hard_calibrate(means, counts, edges):
    """Project empirical leaf means onto the hard order constraints."""
    means = np.asarray(means, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if means.ndim != 1 or counts.ndim != 1 or len(means) != len(counts):
        raise ValueError("means and counts must be one-dimensional arrays of equal length")
    if np.any(counts <= 0.0):
        raise ValueError("all leaf counts must be positive")

    edges = list(edges)
    if not edges:
        return means.copy()

    k = len(means)
    B = np.zeros((len(edges), k), dtype=float)
    for r, (a, b) in enumerate(edges):
        B[r, a] = 1.0
        B[r, b] = -1.0

    constraint = LinearConstraint(B, -np.inf, np.zeros(len(edges)))
    x0 = np.repeat(np.average(means, weights=counts), k)

    def objective(c):
        return 0.5 * np.sum(counts * (c - means) ** 2)

    def gradient(c):
        return counts * (c - means)

    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        constraints=[constraint],
        options={"maxiter": 2000, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(f"hard leaf calibration failed: {result.message}")
    return np.asarray(result.x, dtype=float)


class UncertainExpertRegressionTree:
    """A compact CART-style regression tree used in the experiments."""

    def __init__(
        self,
        max_depth=4,
        min_samples_leaf=15,
        max_thresholds=32,
        min_gain=1e-8,
        split_lambda=0.0,
        leaf_lambda=0.0,
        hard_order=False,
    ):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_thresholds = int(max_thresholds)
        self.min_gain = float(min_gain)
        self.split_lambda = float(split_lambda)
        self.leaf_lambda = float(leaf_lambda)
        self.hard_order = bool(hard_order)
        if self.max_depth < 0 or self.min_samples_leaf < 1 or self.max_thresholds < 1:
            raise ValueError("invalid tree-control parameter")
        if self.split_lambda < 0.0 or self.leaf_lambda < 0.0:
            raise ValueError("regularization parameters must be nonnegative")

    def fit(self, X, y, expert_directions=None, expert_confidences=None):
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y, dtype=float)
        if self.X_.ndim != 2 or self.y_.ndim != 1 or len(self.X_) != len(self.y_):
            raise ValueError("X and y have incompatible shapes")
        if len(self.y_) == 0 or not np.isfinite(self.X_).all() or not np.isfinite(self.y_).all():
            raise ValueError("X and y must be nonempty and finite")

        p = self.X_.shape[1]
        self.directions_ = (
            np.zeros(p, dtype=int)
            if expert_directions is None
            else np.asarray(expert_directions, dtype=int)
        )
        self.confidences_ = (
            np.zeros(p, dtype=float)
            if expert_confidences is None
            else np.asarray(expert_confidences, dtype=float)
        )
        if len(self.directions_) != p or len(self.confidences_) != p:
            raise ValueError("expert arrays must have one entry per predictor")
        if not np.isin(self.directions_, (-1, 0, 1)).all():
            raise ValueError("expert directions must contain only -1, 0, +1")
        if not np.isfinite(self.confidences_).all() or np.any(self.confidences_ < 0.0) or np.any(self.confidences_ > 1.0):
            raise ValueError("expert confidences must lie in [0, 1]")

        self.root_ = self._grow(np.arange(len(self.y_)), depth=0)
        self.leaves_ = []
        self._assign_leaf_ids(self.root_)
        self.edge_weights_, self.hard_edges_ = self._build_leaf_relations()

        means = np.array([leaf.value for leaf in self.leaves_])
        counts = np.array([leaf.n_samples for leaf in self.leaves_], dtype=float)
        if self.hard_order:
            fitted = hard_calibrate(means, counts, self.hard_edges_)
        else:
            fitted = soft_calibrate(means, counts, self.edge_weights_, self.leaf_lambda)
        for leaf, value in zip(self.leaves_, fitted):
            leaf.calibrated_value = float(value)
        return self

    def predict(self, X):
        X = self._validate_prediction_X(X)
        out = np.empty(len(X), dtype=float)
        for i, row in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                node = node.left if row[node.feature] <= node.threshold else node.right
            out[i] = node.calibrated_value
        return out

    def apply(self, X):
        X = self._validate_prediction_X(X)
        out = np.empty(len(X), dtype=int)
        for i, row in enumerate(X):
            node = self.root_
            while not node.is_leaf:
                node = node.left if row[node.feature] <= node.threshold else node.right
            out[i] = node.leaf_id
        return out

    def _validate_prediction_X(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2 or X.shape[1] != self.X_.shape[1]:
            raise ValueError("X has the wrong number of predictors")
        if not np.isfinite(X).all():
            raise ValueError("X must be finite")
        return X

    def _candidate_thresholds(self, values):
        unique = np.unique(values)
        if len(unique) < 2:
            return np.empty(0)
        mids = 0.5 * (unique[:-1] + unique[1:])
        if len(mids) <= self.max_thresholds:
            return mids
        idx = np.linspace(0, len(mids) - 1, self.max_thresholds).round().astype(int)
        return np.unique(mids[idx])

    def _grow(self, indices, depth):
        y = self.y_[indices]
        mean = float(np.mean(y))
        node = Node(indices=indices, value=mean, n_samples=len(indices))
        if depth >= self.max_depth or len(indices) < 2 * self.min_samples_leaf:
            return node

        parent_sse = float(np.sum((y - mean) ** 2))
        if parent_sse <= EPS:
            return node

        best = self._best_split(indices, parent_sse)
        if best is None or best[-1] <= self.min_gain:
            return node

        feature, threshold, left_idx, right_idx, _ = best
        node.feature = feature
        node.threshold = threshold
        node.left = self._grow(left_idx, depth + 1)
        node.right = self._grow(right_idx, depth + 1)
        return node

    def _best_split(self, indices, parent_sse):
        X = self.X_[indices]
        y = self.y_[indices]
        best = None
        best_gain = -np.inf

        for j in range(X.shape[1]):
            direction = int(self.directions_[j])
            confidence = float(self.confidences_[j])
            gamma = self.split_lambda * confidence

            for threshold in self._candidate_thresholds(X[:, j]):
                left_mask = X[:, j] <= threshold
                n_left = int(left_mask.sum())
                n_right = len(y) - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                y_left = y[left_mask]
                y_right = y[~left_mask]
                mean_left = float(np.mean(y_left))
                mean_right = float(np.mean(y_right))
                d = direction * (mean_left - mean_right)
                if self.hard_order and direction != 0 and d > EPS:
                    continue

                sse_left = float(np.sum((y_left - mean_left) ** 2))
                sse_right = float(np.sum((y_right - mean_right) ** 2))
                g0 = 0.5 * (parent_sse - sse_left - sse_right)
                h = n_left * n_right / (n_left + n_right)
                gain = g0 - penalty_increment(d, h, gamma)

                # Deterministic tie breaking: feature order, then threshold order.
                if gain > best_gain + EPS:
                    best_gain = gain
                    best = (
                        j,
                        float(threshold),
                        indices[left_mask],
                        indices[~left_mask],
                        float(gain),
                    )
        return best

    def _assign_leaf_ids(self, node):
        if node.is_leaf:
            node.leaf_id = len(self.leaves_)
            node.calibrated_value = node.value
            self.leaves_.append(node)
            return
        if node.left is None or node.right is None:
            raise RuntimeError("malformed tree: internal node has only one child")
        self._assign_leaf_ids(node.left)
        self._assign_leaf_ids(node.right)

    def _leaf_ids(self, node):
        if node.is_leaf:
            return [node.leaf_id]
        return self._leaf_ids(node.left) + self._leaf_ids(node.right)

    def _build_leaf_relations(self):
        soft_weights = {}
        hard_edges = set()

        def visit(node):
            if node.is_leaf:
                return
            direction = int(self.directions_[node.feature])
            confidence = float(self.confidences_[node.feature])
            if direction != 0:
                left_ids = self._leaf_ids(node.left)
                right_ids = self._leaf_ids(node.right)
                pairs = [(a, b) for a in left_ids for b in right_ids]
                if direction == -1:
                    pairs = [(b, a) for a, b in pairs]
                contribution = confidence / len(pairs) if confidence > 0.0 else 0.0
                for edge in pairs:
                    hard_edges.add(edge)
                    if contribution > 0.0:
                        # Each unordered pair of terminal leaves is separated by
                        # exactly one internal node (their lowest common ancestor).
                        soft_weights[edge] = contribution
            visit(node.left)
            visit(node.right)

        visit(self.root_)
        return soft_weights, sorted(hard_edges)
