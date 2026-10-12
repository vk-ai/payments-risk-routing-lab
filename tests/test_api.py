from fastapi.testclient import TestClient

from prrl import data, features as F, pipeline as P
from prrl.api import create_app


def _scorer(synth):
    tr, va, te = data.time_split(synth)
    kb, fe, enc, tr, va, te = P.build_frames(tr, va, te)
    cols = F.BASE_COLS + F.engineered_cols()
    m = P.fit_model(tr[cols], tr.Class, max_iter=20)
    return P.Scorer(kb, fe, enc, m, cols)


def test_score_endpoint(synth):
    app = create_app(_scorer(synth), threshold=0.5)
    with TestClient(app) as c:
        assert c.get("/health").json() == {"ok": True}
        r = c.post("/score", json={"Time": 100.0, "Amount": 9.99, "V": [0.1] * 28})
        assert r.status_code == 200
        body = r.json()
        assert 0 <= body["fraud_score"] <= 1 and body["decision"] in {"approve", "decline"}
        assert c.post("/score", json={"Time": 1, "Amount": 1, "V": [0.0] * 3}).status_code == 422


def test_online_velocity_increments(synth):
    from prrl.serving import OnlineState
    s = OnlineState()
    k = {"amt_key": 999, "vbin_key": 1}
    assert s.velocity_and_record(k, 0)["vel_amt_key_1h"] == 0
    assert s.velocity_and_record(k, 10)["vel_amt_key_1h"] == 1
    v = s.velocity_and_record(k, 4000)
    assert v["vel_amt_key_1h"] == 0 and v["vel_amt_key_24h"] == 2
