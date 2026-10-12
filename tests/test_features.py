import numpy as np
import pandas as pd
import pytest

from prrl import data, features as F, pipeline as P


def test_time_split_is_ordered(synth):
    tr, va, te = data.time_split(synth)
    assert tr.Time.max() <= va.Time.min() and va.Time.max() <= te.Time.min()
    assert len(tr) + len(va) + len(te) == len(synth)


def test_time_split_rejects_unsorted(synth):
    with pytest.raises(ValueError):
        data.time_split(synth.iloc[::-1])


def test_velocity_counts_only_past_within_window():
    df = pd.DataFrame({"Time": [0, 10, 20, 5000, 5001], "k": [1, 1, 2, 1, 1]})
    assert F.velocity(df, "k", 3600).tolist() == [0, 1, 0, 0, 1]
    assert F.velocity(df, "k", 86400).tolist() == [0, 1, 0, 2, 3]


def test_velocity_ties_use_row_order():
    df = pd.DataFrame({"Time": [5, 5, 5], "k": [7, 7, 7]})
    assert F.velocity(df, "k", 60).tolist() == [0, 1, 2]


def test_target_encoding_excludes_current_and_future_rows():
    df = pd.DataFrame({"amt_key": [1, 1, 1, 1], "vbin_key": [0, 0, 0, 0], "Class": [1, 0, 0, 1]})
    enc = F.TimeRespectingTargetEncoder(smoothing=0.0, keys=("amt_key",))
    out = enc.fit_transform(df)
    # row0 has no history -> 0/0 = nan with m=0; rows 1..3 see only earlier labels
    assert np.isnan(out.te_amt_key.iloc[0])
    assert out.te_amt_key.iloc[1:].tolist() == [1.0, 0.5, 1 / 3]
    # changing a FUTURE label must not change earlier encodings
    df2 = df.copy(); df2.loc[3, "Class"] = 0
    out2 = enc.fit_transform(df2)
    assert out2.te_amt_key.iloc[1:].tolist() == out.te_amt_key.iloc[1:].tolist()


def test_target_encoding_frozen_for_later_frames():
    tr = pd.DataFrame({"amt_key": [1, 1, 2], "vbin_key": [0, 0, 0], "Class": [1, 1, 0]})
    enc = F.TimeRespectingTargetEncoder(smoothing=1.0)
    enc.fit_transform(tr)
    later = pd.DataFrame({"amt_key": [1, 99], "vbin_key": [0, 0], "Class": [0, 1]})
    a = enc.transform(later)
    later.loc[:, "Class"] = [1, 0]
    b = enc.transform(later)
    assert a.te_amt_key.tolist() == b.te_amt_key.tolist()  # labels of later frames unused
    assert a.te_amt_key.iloc[1] == pytest.approx(enc.prior)


def test_frequency_encoder_unseen_is_zero():
    fe = F.FrequencyEncoder().fit(pd.DataFrame({"amt_key": [1, 1, 2], "vbin_key": [0, 0, 0]}))
    out = fe.transform(pd.DataFrame({"amt_key": [1, 3], "vbin_key": [0, 5]}))
    assert out.freq_amt_key.tolist() == [2.0, 0.0] and out.freq_vbin_key.tolist() == [3.0, 0.0]


def test_build_frames_and_fit(synth):
    tr, va, te = data.time_split(synth)
    kb, fe, enc, tr, va, te = P.build_frames(tr, va, te)
    cols = F.BASE_COLS + F.engineered_cols()
    assert not tr[cols].isna().any().any()
    m = P.fit_model(tr[cols], tr.Class, max_iter=20)
    met, p = P.evaluate(m, te[cols], te.Class)
    assert 0 <= met["pr_auc"] <= 1 and len(p) == len(te)


def test_checksum_detects_mismatch(tmp_path):
    f = tmp_path / "x.csv"; f.write_text("a")
    assert data.sha256(f) != data.DATA_SHA256
