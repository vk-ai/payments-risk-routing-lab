"""Online feature computation mirroring the offline features (backward-only)."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from . import features as F


class OnlineState:
    """Keeps recent event times per key for velocity features. In-memory, single process."""

    def __init__(self):
        self.hist = {k: defaultdict(deque) for k in F.KEYS}
        self.glob = deque()

    def velocity_and_record(self, keys: dict, t: float) -> dict:
        out = {}
        for k in F.KEYS:
            dq = self.hist[k][keys[k]]
            while dq and dq[0] < t - max(F.WINDOWS.values()):
                dq.popleft()
            for name, w in F.WINDOWS.items():
                out[f"vel_{k}_{name}"] = sum(1 for x in dq if x >= t - w)
            dq.append(t)
        while self.glob and self.glob[0] < t - 3600:
            self.glob.popleft()
        out["vel_global_1h"] = len(self.glob)
        self.glob.append(t)
        return out


def featurize_one(scorer, state: OnlineState, txn: dict) -> np.ndarray:
    """Same features as the offline pipeline, computed with plain dict/numpy ops (fast path).
    Returns a (1, n_features) array in scorer.cols order."""
    kb = scorer.keys
    feats = {f"V{i}": float(txn[f"V{i}"]) for i in range(1, 29)}
    feats["Amount"] = float(txn["Amount"])
    feats["hour"] = float((float(txn["Time"]) // 3600) % 24)
    code = 0
    for c in kb.cols:
        code = code * kb.q + int(np.searchsorted(kb.edges[c], feats[c]))
    keys = {"amt_key": int(np.round(feats["Amount"] * 100)), "vbin_key": code}
    feats.update(state.velocity_and_record(keys, float(txn["Time"])))
    for k in F.KEYS:
        feats[f"freq_{k}"] = float(scorer.freq.maps[k].get(keys[k], 0))
        feats[f"te_{k}"] = float(scorer.te.stats[k].get(keys[k], scorer.te.prior))
    return np.array([[feats[c] for c in scorer.cols]], dtype=np.float64)
