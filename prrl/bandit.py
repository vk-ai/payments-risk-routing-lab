"""Contextual-bandit acquirer routing, evaluated by offline replay on SIMULATED outcomes.

Everything about acquirers here is made up: the per-acquirer approval curves and fees are a
toy simulator, not real acquirer data. The context features (amount, fraud score, segment)
come from the real transactions.
"""
from __future__ import annotations

import numpy as np


class AcquirerSim:
    """K simulated acquirers. P(auth approved | x, a) = sigmoid(x @ w_a); fee_a as fraction."""

    def __init__(self, d: int, k: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        # Acquirers "specialise": strong per-segment and amount effects, similar intercepts,
        # so the best acquirer depends on context (otherwise routing is trivial).
        self.w = rng.normal(0, 1.5, size=(k, d))
        self.w[:, 0] = 0.6 + rng.normal(0, 0.1, size=k)
        if d > 2:
            self.w[:, 2] = -np.abs(rng.normal(1.0, 0.5, size=k))  # riskier txns get approved less
        self.fee = np.linspace(0.010, 0.025, k)
        self.k, self.d = k, d

    def p_approve(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(x @ self.w.T)))  # (n, k)

    def reward(self, x, a, rng):
        p = self.p_approve(x[None, :])[0, a]
        ok = rng.random() < p
        return float(ok) * (1.0 - 10 * self.fee[a])  # approved revenue share net of (scaled) fee

    def expected_reward(self, X):
        return self.p_approve(X) * (1.0 - 10 * self.fee)[None, :]


def make_context(amount, score, segment, n_seg: int = 4) -> np.ndarray:
    """[1, log-amount (scaled), fraud score, one-hot segment]."""
    amount, score, segment = map(np.asarray, (amount, score, segment))
    la = (np.log1p(amount) - 3.0) / 2.0
    oh = np.eye(n_seg)[segment % n_seg]
    return np.column_stack([np.ones_like(la), la, score, oh])


def log_uniform(X, sim: AcquirerSim, seed: int = 1):
    """Logging policy: uniform random routing (propensity 1/K)."""
    rng = np.random.default_rng(seed)
    A = rng.integers(0, sim.k, size=len(X))
    R = np.array([sim.reward(x, a, rng) for x, a in zip(X, A)])
    return A, R, np.full(len(X), 1.0 / sim.k)


class LinUCB:
    def __init__(self, k, d, alpha=0.5, lam=1.0):
        self.A = np.stack([lam * np.eye(d)] * k)
        self.b = np.zeros((k, d))
        self.alpha = alpha

    def select(self, x):
        Ainv = np.linalg.inv(self.A)
        theta = np.einsum("kij,kj->ki", Ainv, self.b)
        ucb = theta @ x + self.alpha * np.sqrt(np.einsum("i,kij,j->k", x, Ainv, x))
        return int(np.argmax(ucb))

    def update(self, x, a, r):
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x


def replay(policy, X, A, R):
    """Li et al. (2011) replay: keep events where policy matches the logged uniform action.
    Unbiased for uniform logging. Returns (mean reward, n_matched)."""
    rewards = []
    for x, a, r in zip(X, A, R):
        choice = policy.select(x)
        if choice == a:
            policy.update(x, a, r)
            rewards.append(r)
    return (float(np.mean(rewards)) if rewards else float("nan")), len(rewards)


def ips_value(choices, A, R, prop):
    """IPS estimate of a fixed (non-learning) policy's value, with a normal-approx 95% CI."""
    w = (np.asarray(choices) == A) / prop
    v = w * R
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    return m, (m - 1.96 * se, m + 1.96 * se)


class Static:
    def __init__(self, a):
        self.a = a

    def select(self, x):
        return self.a

    def update(self, *a):
        pass


class RandomPolicy:
    def __init__(self, k, seed=12345):
        self.k, self.rng = k, np.random.default_rng(seed)

    def select(self, x):
        return int(self.rng.integers(self.k))

    def update(self, *a):
        pass
