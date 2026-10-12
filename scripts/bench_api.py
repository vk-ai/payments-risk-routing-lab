"""Start uvicorn on the real trained artifact and measure /score latency under concurrent load."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

CONC = int(os.environ.get("CONC", 20)); N = int(os.environ.get("N", 4000)); PORT = 8765


async def main():
    rows = np.load("artifacts/test_sample.npy")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "prrl.api:create_app", "--factory",
                             "--port", str(PORT), "--log-level", "warning", "--workers", "1"])
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}", timeout=10) as c:
            for _ in range(100):
                try:
                    if (await c.get("/health")).status_code == 200:
                        break
                except httpx.HTTPError:
                    await asyncio.sleep(0.2)
            def body(i):
                r = rows[i % len(rows)]
                return {"Time": float(r[0]), "Amount": float(r[1]), "V": [float(x) for x in r[2:]]}
            for i in range(200):  # warmup
                await c.post("/score", json=body(i))
            lat, errs, q = [], 0, iter(range(N))
            async def worker():
                nonlocal errs
                for i in q:
                    t = time.perf_counter()
                    r = await c.post("/score", json=body(i))
                    lat.append((time.perf_counter() - t) * 1000)
                    errs += r.status_code != 200
            t0 = time.perf_counter()
            await asyncio.gather(*[worker() for _ in range(CONC)])
            wall = time.perf_counter() - t0
        res = {"requests": N, "concurrency": CONC, "errors": errs, "throughput_rps": round(N / wall, 1),
               "p50_ms": round(float(np.percentile(lat, 50)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
               "p99_ms": round(float(np.percentile(lat, 99)), 2), "cpu_count": os.cpu_count(),
               "loadavg_at_end": [round(x, 2) for x in os.getloadavg()]}
        Path("results/api_latency.json").write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
    finally:
        proc.terminate(); proc.wait()

asyncio.run(main())
