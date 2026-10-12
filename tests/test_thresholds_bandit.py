import numpy as np

from prrl import bandit as B, thresholds as T


def test_tradeoff_monotone():
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.05).astype(int)
    s = np.clip(y * 0.6 + rng.random(1000) * 0.5, 0, 1)
    amt = rng.uniform(1, 100, 1000)
    tab = T.tradeoff_table(s, y, amt, [0.2, 0.5, 0.8, 1.01])
    assert tab.approval_rate.is_monotonic_increasing
    assert tab.fraud_loss_amount.is_monotonic_increasing
    assert tab.fraud_loss_share.iloc[-1] == 1.0


def test_tradeoff_exact_numbers():
    tab = T.tradeoff_table([0.1, 0.9, 0.6], [0, 1, 1], [10, 20, 30], [0.7])
    r = tab.iloc[0]
    assert r.approval_rate == 2 / 3 and r.fraud_loss_amount == 30 and r.fraud_caught_count == 1


def test_linucb_beats_random_on_sim():
    rng = np.random.default_rng(0)
    X = B.make_context(rng.uniform(1, 500, 6000), rng.random(6000), rng.integers(0, 4, 6000))
    sim = B.AcquirerSim(d=X.shape[1], seed=3)
    A, R, p = B.log_uniform(X, sim, seed=1)
    lin, _ = B.replay(B.LinUCB(sim.k, X.shape[1]), X, A, R)
    rnd, _ = B.replay(B.RandomPolicy(sim.k), X, A, R)
    assert lin > rnd


def test_ips_matches_truth_for_static():
    rng = np.random.default_rng(1)
    X = B.make_context(rng.uniform(1, 500, 20000), rng.random(20000), rng.integers(0, 4, 20000))
    sim = B.AcquirerSim(d=X.shape[1], seed=4)
    A, R, p = B.log_uniform(X, sim, seed=2)
    v, (lo, hi) = B.ips_value(np.zeros(len(X), int), A, R, p)
    truth = sim.expected_reward(X)[:, 0].mean()
    assert lo <= truth <= hi


def test_replay_only_counts_matches():
    X = np.ones((4, 2)); A = np.array([0, 1, 0, 1]); R = np.array([1.0, 0.0, 1.0, 0.0])
    v, n = B.replay(B.Static(0), X, A, R)
    assert n == 2 and v == 1.0
