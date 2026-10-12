import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synth():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({f"V{i}": rng.normal(size=n) for i in range(1, 29)})
    df["Time"] = np.sort(rng.uniform(0, 3 * 86400, n)).round()
    df["Amount"] = rng.choice([1.0, 9.99, 25.0, 100.0, 3.5], n)
    df["Class"] = ((df.V1 > 1.5) | (rng.random(n) < 0.02)).astype(int)
    df = df.reset_index(names="row_id")
    return df
