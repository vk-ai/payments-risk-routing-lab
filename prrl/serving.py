"""Online feature computation mirroring the offline features (backward-only)."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

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


def featurize_one(scorer, state: OnlineState, txn: dict) -> pd.DataFrame:
    df = pd.DataFrame([txn])
    df = F.add_basic(scorer.keys.transform(df))
    vel = state.velocity_and_record({k: df[k].iloc[0] for k in F.KEYS}, float(txn["Time"]))
    for c, v in vel.items():
        df[c] = v
    df = scorer.freq.transform(scorer.te.transform(df))
    return df[scorer.cols].astype(np.float64)
