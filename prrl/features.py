"""High-cardinality feature engineering that only looks backwards in time.

The ULB dataset has no card/merchant/device IDs, so we derive two *proxy* keys:
- amt_key:  the exact transaction amount in cents (~30k distinct values). A crude stand-in
            for a "price point / merchant SKU" entity.
- vbin_key: decile bins of the PCA components V1..V4 combined (up to 10^4 cells). A crude
            stand-in for a "customer segment / device cluster" entity.
These are NOT real entities; velocity on them means "how many similar-looking transactions
happened recently", not "how many times this card was used".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ("amt_key", "vbin_key")
WINDOWS = {"1h": 3600, "24h": 86400}


class KeyBuilder:
    """Fits V1..V4 decile edges on train only."""

    def __init__(self, cols=("V1", "V2", "V3", "V4"), q: int = 10):
        self.cols, self.q, self.edges = cols, q, {}

    def fit(self, df: pd.DataFrame) -> "KeyBuilder":
        qs = np.linspace(0, 1, self.q + 1)[1:-1]
        self.edges = {c: np.quantile(df[c].to_numpy(), qs) for c in self.cols}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["amt_key"] = np.round(out["Amount"].to_numpy() * 100).astype(np.int64)
        code = np.zeros(len(out), dtype=np.int64)
        for c in self.cols:
            code = code * self.q + np.searchsorted(self.edges[c], out[c].to_numpy())
        out["vbin_key"] = code
        return out


def velocity(df: pd.DataFrame, key: str, window: int) -> np.ndarray:
    """Number of *earlier* rows with the same key within `window` seconds (current row excluded).

    Uses only Time and key (no labels), so it is safe to compute across the whole stream
    as long as rows are in time order.
    """
    t = df["Time"].to_numpy(dtype=np.float64)
    codes, _ = pd.factorize(df[key], sort=True)
    order = np.lexsort((np.arange(len(df)), t, codes))  # by code, time, original position
    big = t.max() + window + 1.0
    comp = codes[order] * big + t[order]
    left = np.searchsorted(comp, comp - window, side="left")
    cnt_sorted = np.arange(len(comp)) - left
    out = np.empty(len(df), dtype=np.int64)
    out[order] = cnt_sorted
    return out


def add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for k in KEYS:
        for name, w in WINDOWS.items():
            out[f"vel_{k}_{name}"] = velocity(out, k, w)
    # global traffic in the last hour (a load proxy), backward-only
    t = out["Time"].to_numpy()
    out["vel_global_1h"] = np.arange(len(t)) - np.searchsorted(t, t - 3600, side="left")
    return out


class FrequencyEncoder:
    """Counts per key value, fitted on train only; unseen values -> 0."""

    def fit(self, df, keys=KEYS):
        self.maps = {k: df[k].value_counts().to_dict() for k in keys}
        return self

    def transform(self, df):
        out = df.copy()
        for k, m in self.maps.items():
            out[f"freq_{k}"] = out[k].map(m).fillna(0).astype(np.float64)
        return out


class TimeRespectingTargetEncoder:
    """Smoothed target mean per key.

    - On the training frame (time-ordered), each row only sees labels of *earlier* rows
      with the same key (expanding mean, current row excluded) -> no self-leakage.
    - For later frames (valid/test/serving), statistics are frozen at the end of train,
      i.e. we assume labels for post-train traffic are not yet available (chargeback delay).
    """

    def __init__(self, smoothing: float = 20.0, keys=KEYS):
        self.m, self.keys = smoothing, keys

    def fit_transform(self, train: pd.DataFrame, y: str = "Class") -> pd.DataFrame:
        out = train.copy()
        self.prior = float(out[y].mean())
        self.stats = {}
        for k in self.keys:
            g = out.groupby(k, sort=False)[y]
            prev_sum = g.cumsum() - out[y]
            prev_cnt = g.cumcount()
            out[f"te_{k}"] = (prev_sum + self.m * self.prior) / (prev_cnt + self.m)
            agg = out.groupby(k)[y].agg(["sum", "count"])
            self.stats[k] = ((agg["sum"] + self.m * self.prior) / (agg["count"] + self.m)).to_dict()
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for k in self.keys:
            out[f"te_{k}"] = out[k].map(self.stats[k]).fillna(self.prior).astype(np.float64)
        return out


BASE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "hour"]


def add_basic(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = (out["Time"] // 3600) % 24
    return out


def engineered_cols() -> list[str]:
    cols = [f"vel_{k}_{n}" for k in KEYS for n in WINDOWS] + ["vel_global_1h"]
    cols += [f"freq_{k}" for k in KEYS] + [f"te_{k}" for k in KEYS]
    return cols
