"""Approval-rate vs fraud-loss tradeoffs (amount weighted)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tradeoff_table(score, y, amount, thresholds) -> pd.DataFrame:
    score, y, amount = map(np.asarray, (score, y, amount))
    total_fraud_amt = float(amount[y == 1].sum())
    rows = []
    for t in thresholds:
        approve = score < t  # decline when score >= t
        fraud_approved = approve & (y == 1)
        legit_declined = (~approve) & (y == 0)
        rows.append({
            "threshold": float(t),
            "approval_rate": float(approve.mean()),
            "fraud_loss_amount": float(amount[fraud_approved].sum()),
            "fraud_loss_share": float(amount[fraud_approved].sum() / total_fraud_amt) if total_fraud_amt else 0.0,
            "fraud_caught_count": int(((~approve) & (y == 1)).sum()),
            "false_declines": int(legit_declined.sum()),
            "false_decline_amount": float(amount[legit_declined].sum()),
        })
    return pd.DataFrame(rows)


def threshold_for_approval(score, target_rate: float) -> float:
    """Threshold that approves ~target_rate of traffic."""
    return float(np.quantile(np.asarray(score), target_rate))
