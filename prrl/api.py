"""FastAPI scoring endpoint. Loads artifacts/scorer.joblib (or PRRL_ARTIFACT)."""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .serving import OnlineState, featurize_one

class Txn(BaseModel):
    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)
    V: list[float] = Field(min_length=28, max_length=28, description="PCA features V1..V28")


def create_app(scorer=None, threshold: float | None = None) -> FastAPI:
    state = {"scorer": scorer, "thr": threshold, "online": OnlineState(), "lock": threading.Lock()}

    @asynccontextmanager
    async def lifespan(_app):
        if state["scorer"] is None:
            art = joblib.load(os.environ.get("PRRL_ARTIFACT", "artifacts/scorer.joblib"))
            state["scorer"], state["thr"] = art["scorer"], art["threshold"]
        yield

    app = FastAPI(title="payments-risk-routing-lab scorer", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"ok": state["scorer"] is not None}

    @app.post("/score")
    def score(t: Txn):
        txn = {"Time": t.Time, "Amount": t.Amount, **{f"V{i+1}": v for i, v in enumerate(t.V)}}
        with state["lock"]:
            X = featurize_one(state["scorer"], state["online"], txn)
        p = float(state["scorer"].model.predict_proba(X)[0, 1])
        thr = state["thr"] if state["thr"] is not None else 0.5
        return {"fraud_score": p, "decision": "decline" if p >= thr else "approve", "threshold": thr}

    return app
