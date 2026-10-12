"""Dataset loading, checksum pinning and leakage-safe time split."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

# ULB "Credit Card Fraud Detection" (Worldline + ULB MLG), public mirror used by the TensorFlow docs.
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DATA_SHA256 = "76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL, path)
    got = sha256(path)
    if got != DATA_SHA256:
        raise ValueError(f"checksum mismatch for {path}: {got}")
    return path


def load(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    # stable time order; keep original row id as tiebreaker
    df = df.reset_index(names="row_id").sort_values(["Time", "row_id"], kind="stable")
    return df.reset_index(drop=True)


def time_split(df: pd.DataFrame, train: float = 0.6, valid: float = 0.2):
    """Split strictly by time: earliest rows train, then valid, then test. No shuffling."""
    if not df["Time"].is_monotonic_increasing:
        raise ValueError("frame must be sorted by Time")
    n = len(df)
    a, b = int(n * train), int(n * (train + valid))
    return df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()
