"""Web server for the React frontend: a small JSON API plus the built static site.

Standard library only (no Flask/FastAPI needed). Runs from saved checkpoints, never trains.

    python server.py                 # http://127.0.0.1:8000  (serves frontend/dist + /api)
    python server.py --port 8080

API (all GET, JSON):
    /api/summary                     every results/*.csv as rows, run metadata, the computed conclusion
    /api/layout?d=3                  lattice geometry: data qubits, check (ancilla) positions, detector coords
    /api/shot?d=3&p=0.01&seed=4000   sample one shot, decode with PyMatching / MLP / CNN, matching edges
    /api/benchmark?d=3&p=0.01&shots=20000   live LER of every decoder on fresh shots, with Wilson CIs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import torch

import config
import make_report
from src import mwpm
from src.circuits import circuit_stats, make_circuit
from src.data import sample
from src.evaluate import wilson
from src.train import load_trained, predict_logits

DIST = config.ROOT / "frontend" / "dist"
MAX_BENCH_SHOTS = 200_000
P_MIN, P_MAX = 0.0005, 0.03
LOCK = threading.Lock()  # one decode at a time keeps torch thread use predictable


# --------------------------------------------------------------------------- cached heavy objects
@lru_cache(maxsize=64)
def circuit_for(d: int, p: float):
    return make_circuit(d, p)


@lru_cache(maxsize=64)
def matcher_for(d: int, p: float):
    return mwpm.build_matcher(circuit_for(d, p))


@lru_cache(maxsize=None)
def models_for(d: int) -> dict:
    return {"mlp": load_trained("mlp", d), "cnn": load_trained("cnn", d)}


# --------------------------------------------------------------------------- handlers
def _clean(v):
    """JSON-safe scalars (NaN -> None, numpy -> python)."""
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(float(v)) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path, comment="#")
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def summary() -> dict:
    R = config.RESULTS
    names = ["ler_results", "throughput", "model_sizes", "disagreement", "generalisation", "threshold"]
    out = {n: _rows(R / f"{n}.csv") for n in names}
    out["train_logs"] = {p.stem.removeprefix("train_log_"): _rows(p) for p in sorted(R.glob("train_log_*.csv"))}
    read_json = lambda p: json.loads(p.read_text()) if p.exists() else {}
    out["eval_meta"] = read_json(R / "eval_meta.json")
    out["runtime"] = read_json(R / "runtime.json")
    out["checkpoints"] = {p.stem: read_json(p) for p in sorted(config.CHECKPOINTS.glob("*.json"))}
    try:
        out["conclusion"] = make_report.conclusion_lines()
    except Exception as e:  # results missing or partial
        out["conclusion"] = [f"(conclusion unavailable: {e})"]
    out["config"] = {"distances": config.DISTANCES, "p_sweep": config.P_SWEEP, "fixed_p": config.FIXED_P,
                     "seeds": config.SEEDS, "names": make_report.NAMES}
    return out


def layout(d: int) -> dict:
    c = circuit_for(d, 0.0)
    coords = c.get_detector_coordinates()
    dets = [list(map(float, coords[i])) for i in sorted(coords)]
    anc = sorted({(x, y) for x, y, _ in dets})
    z_type = {(x, y) for x, y, t in dets if t == 0}  # Z-memory: only Z checks are deterministic at t=0
    data = sorted({(x, y) for x, y in c.get_final_qubit_coordinates().values() if x % 2 == 1 and y % 2 == 1})
    return {"d": d, "rounds": config.rounds(d), "stats": circuit_stats(c),
            "data": [list(q) for q in data],
            "checks": [{"x": x, "y": y, "type": "Z" if (x, y) in z_type else "X"} for x, y in anc],
            "detectors": dets}


def shot(d: int, p: float, seed: int, nonempty: bool) -> dict:
    c = circuit_for(d, p)
    for s in range(seed, seed + 500):  # skip silent shots if asked: they are boring to look at
        det, obs = sample(c, 1, seed=s)
        if not nonempty or det[0].any():
            break
    row, truth = det[0], bool(obs[0])
    m = matcher_for(d, p)
    t0 = time.perf_counter()
    pm = bool(mwpm.decode(m, row[None])[0])
    t_pm = time.perf_counter() - t0
    edges = m.decode_to_edges_array(row.astype(np.uint8)).tolist()
    out = {"d": d, "p": p, "seed": s, "fired": np.flatnonzero(row).tolist(), "truth": truth,
           "edges": edges, "decoders": {"pymatching": {"flip": pm, "us": t_pm * 1e6, "prob": None}}}
    for name, model in models_for(d).items():
        t0 = time.perf_counter()
        logit = float(predict_logits(model, row[None])[0])
        out["decoders"][name] = {"flip": logit > 0, "us": (time.perf_counter() - t0) * 1e6,
                                 "prob": 1 / (1 + math.exp(-logit))}
    return out


def benchmark(d: int, p: float, shots: int, seed: int) -> dict:
    c = circuit_for(d, p)
    det, obs = sample(c, shots, seed=seed)
    res = {}

    def add(name: str, pred: np.ndarray, secs: float | None):
        err = int((pred != obs).sum())
        lo, hi = wilson(err, shots)
        res[name] = {"errors": err, "ler": err / shots, "ci_low": lo, "ci_high": hi,
                     "us_per_shot": None if secs is None else secs / shots * 1e6}

    add("no_decoder", np.zeros_like(obs), None)
    m = matcher_for(d, p)
    t0 = time.perf_counter()
    add("pymatching", mwpm.decode(m, det), time.perf_counter() - t0)
    for name, model in models_for(d).items():
        t0 = time.perf_counter()
        pred = predict_logits(model, det) > 0
        add(name, pred, time.perf_counter() - t0)
    return {"d": d, "p": p, "shots": shots, "seed": seed, "results": res}


# --------------------------------------------------------------------------- HTTP plumbing
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DIST), **kw)

    def log_message(self, fmt, *args):  # quieter console: only API calls
        if self.path.startswith("/api"):
            super().log_message(fmt, *args)

    def _json(self, obj, status: int = 200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path.startswith("/api/"):
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                self._json(self._api(url.path, q))
            except (ValueError, KeyError) as e:
                self._json({"error": str(e)}, 400)
            except FileNotFoundError as e:
                self._json({"error": f"missing file: {e}. Run `python run_all.py --quick` first."}, 503)
            return
        if url.path.startswith("/figures/"):  # the matplotlib PNGs, for the report section
            f = config.FIGURES / Path(url.path).name
            if f.is_file():
                body = f.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        if not DIST.exists():
            self._json({"error": "frontend not built. Run `cd frontend && npm run build`."}, 503)
            return
        if not (DIST / url.path.lstrip("/")).is_file():
            self.path = "/index.html"  # single-page app fallback
        super().do_GET()

    def _api(self, path: str, q: dict):
        d = int(q.get("d", 3))
        if path == "/api/summary":
            return summary()
        if d not in config.DISTANCES:
            raise ValueError(f"d must be one of {config.DISTANCES}")
        if path == "/api/layout":
            return layout(d)
        p = round(float(q.get("p", 0.01)), 5)
        if not P_MIN <= p <= P_MAX:
            raise ValueError(f"p must be in [{P_MIN}, {P_MAX}]")
        seed = int(q.get("seed", config.SEEDS["demo"]))
        with LOCK:
            if path == "/api/shot":
                return shot(d, p, seed, q.get("nonempty", "1") == "1")
            if path == "/api/benchmark":
                shots = min(max(int(q.get("shots", 20_000)), 1_000), MAX_BENCH_SHOTS)
                return benchmark(d, p, shots, seed)
        raise KeyError(f"unknown endpoint {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    for d in config.DISTANCES:  # load checkpoints up front so the first click is fast
        models_for(d)
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"QEC decoder UI on http://{a.host}:{a.port}  (Ctrl+C to stop)")
    if not DIST.exists():
        print("  note: frontend/dist missing; build it with `cd frontend && npm run build`, or use `npm run dev`.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
