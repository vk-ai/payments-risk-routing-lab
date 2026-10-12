"""End-to-end experiment on the real ULB data. Writes results/ and artifacts/ (artifacts gitignored)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prrl import bandit as B, data, features as F, pipeline as P, thresholds as T

R = Path("results"); R.mkdir(exist_ok=True)
A = Path("artifacts"); A.mkdir(exist_ok=True)
t0 = time.time()
df = data.load(data.download())
train, valid, test = data.time_split(df)
kb, fe, te, tr, va, ts = P.build_frames(train, valid, test)
out = {"dataset": {"rows": len(df), "fraud_rate": float(df.Class.mean()),
                   "split_rows": [len(tr), len(va), len(ts)],
                   "split_frauds": [int(x.Class.sum()) for x in (tr, va, ts)],
                   "distinct_amt_key": int(df.Amount.round(2).nunique()),
                   "distinct_vbin_key_train": int(tr.vbin_key.nunique())}}
base, eng = F.BASE_COLS, F.BASE_COLS + F.engineered_cols()
res, preds = {}, {}
for name, cols in (("base", base), ("base_plus_engineered", eng)):
    m = P.fit_model(tr[cols], tr.Class)
    res[name] = {"valid": P.evaluate(m, va[cols], va.Class)[0]}
    res[name]["test"], preds[name] = P.evaluate(m, ts[cols], ts.Class)
    preds[name + "_valid"] = m.predict_proba(va[cols])[:, 1]
    if name == "base_plus_engineered":
        model = m
d, ci = P.bootstrap_pr_auc_diff(ts.Class, preds["base"], preds["base_plus_engineered"])
res["test_pr_auc_diff_engineered_minus_base"] = {"mean": d, "ci95": ci}
# ablation: one feature group at a time on top of base
for grp, extra in (("velocity", [c for c in F.engineered_cols() if c.startswith("vel_")]),
                   ("frequency", [c for c in F.engineered_cols() if c.startswith("freq_")]),
                   ("target_enc", [c for c in F.engineered_cols() if c.startswith("te_")])):
    m = P.fit_model(tr[base + extra], tr.Class)
    res[f"ablation_base_plus_{grp}"] = {"test": P.evaluate(m, ts[base + extra], ts.Class)[0]}
out["models"] = res

# thresholds picked on VALID approval-rate quantiles, applied to TEST
score = preds["base_plus_engineered"]
rates = [0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995]
thr = [T.threshold_for_approval(preds["base_plus_engineered_valid"], r) for r in rates]
tab = T.tradeoff_table(score, ts.Class, ts.Amount, thr)
tab.insert(0, "target_approval_rate_on_valid", rates)
tab.to_csv(R / "threshold_tradeoff.csv", index=False)
curve = T.tradeoff_table(score, ts.Class, ts.Amount, np.quantile(score, np.linspace(0.9, 1, 201)))
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(curve.approval_rate * 100, curve.fraud_loss_amount, label="GBDT + engineered")
cb = T.tradeoff_table(preds["base"], ts.Class, ts.Amount, np.quantile(preds["base"], np.linspace(0.9, 1, 201)))
ax.plot(cb.approval_rate * 100, cb.fraud_loss_amount, "--", label="GBDT base")
ax.set_xlabel("approval rate on test (%)"); ax.set_ylabel("fraud loss (approved fraud amount, EUR)")
ax.set_title("ULB test split: approval rate vs fraud loss"); ax.legend(); fig.tight_layout()
fig.savefig(R / "threshold_tradeoff.png", dpi=120)
out["fraud_amount_test_total"] = float(ts.Amount[ts.Class == 1].sum())
out["threshold_table"] = tab.to_dict(orient="records")
serve_thr = thr[rates.index(0.995)]

# bandit routing on SIMULATED acquirers
def ctx(frame, s):
    return B.make_context(frame.Amount.to_numpy(), s, frame.vbin_key.to_numpy())
Xv, Xt = ctx(va, preds["base_plus_engineered_valid"]), ctx(ts, score)
sim = B.AcquirerSim(d=Xt.shape[1], k=3, seed=0)
Av, Rv, pv = B.log_uniform(Xv, sim, seed=1)
At, Rt, pt = B.log_uniform(Xt, sim, seed=2)
ips_static = {a: B.ips_value(np.full(len(Xv), a), Av, Rv, pv)[0] for a in range(sim.k)}
best_static = max(ips_static, key=ips_static.get)
E = sim.expected_reward(Xt)
band = {"note": "acquirer outcomes are SIMULATED (toy logistic model + made-up fees)",
        "k_acquirers": sim.k, "logged_events_test": len(Xt), "best_static_chosen_on_valid": int(best_static)}
for name, pol in (("random", B.RandomPolicy(sim.k)), ("static_best", B.Static(best_static)),
                  ("linucb", B.LinUCB(sim.k, Xt.shape[1], alpha=0.5))):
    v, n = B.replay(pol, Xt, At, Rt)
    band[name] = {"replay_value": v, "replay_matched": n}
    if name == "linucb":
        lin = pol
for name, ch in (("static_best", np.full(len(Xt), best_static)),):
    m, ci = B.ips_value(ch, At, Rt, pt); band[name]["ips_value"] = m; band[name]["ips_ci95"] = ci
lin_choices = np.array([lin.select(x) for x in Xt])
band["true_expected_reward"] = {
    "random": float(E.mean()), "static_best": float(E[:, best_static].mean()),
    "linucb_final_frozen": float(E[np.arange(len(Xt)), lin_choices].mean()),
    "oracle": float(E.max(1).mean())}
out["bandit"] = band
out["runtime_s"] = round(time.time() - t0, 1)
(R / "metrics.json").write_text(json.dumps(out, indent=2))
scorer = P.Scorer(kb, fe, te, model, eng)
joblib.dump({"scorer": scorer, "threshold": serve_thr}, A / "scorer.joblib")
np.save(A / "test_sample.npy", ts[["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]].head(5000).to_numpy())
print(json.dumps(out, indent=2))
