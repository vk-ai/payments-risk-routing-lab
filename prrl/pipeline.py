"""Build features for train/valid/test and fit the scorer."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from . import features as F


@dataclass
class Scorer:
    keys: F.KeyBuilder
    freq: F.FrequencyEncoder
    te: F.TimeRespectingTargetEncoder
    model: HistGradientBoostingClassifier
    cols: list = field(default_factory=list)


def build_frames(train, valid, test):
    kb = F.KeyBuilder().fit(train)
    full = pd.concat([train, valid, test], keys=["train", "valid", "test"], names=["split", None])
    full = F.add_basic(kb.transform(full.reset_index(level=0)))
    full = F.add_velocity(full)  # backward-only, label-free
    tr, va, te_ = (full[full.split == s].drop(columns="split") for s in ("train", "valid", "test"))
    fe = F.FrequencyEncoder().fit(tr)
    tenc = F.TimeRespectingTargetEncoder()
    tr = fe.transform(tenc.fit_transform(tr))
    va, te_ = (fe.transform(tenc.transform(x)) for x in (va, te_))
    return kb, fe, tenc, tr, va, te_


def fit_model(X, y, seed=0, max_iter=300):
    m = HistGradientBoostingClassifier(
        max_iter=max_iter, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
        early_stopping=False, random_state=seed)
    return m.fit(X, y)


def evaluate(model, X, y):
    p = model.predict_proba(X)[:, 1]
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p))}, p


def bootstrap_pr_auc_diff(y, p1, p2, n=200, seed=0):
    rng = np.random.default_rng(seed)
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        d.append(average_precision_score(y[i], p2[i]) - average_precision_score(y[i], p1[i]))
    d = np.array(d)
    return float(d.mean()), (float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)))
