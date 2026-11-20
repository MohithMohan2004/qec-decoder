"""Evaluation on the fixed test sets: LER with Wilson CIs, timing, disagreement,
generalisation, threshold estimate and the sanity checks that gate the pipeline.

CLI:  python -m src.evaluate [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import time

import numpy as np
import pandas as pd
import torch

import config
from src import mwpm
from src.circuits import make_circuit
from src.data import fixed_set
from src.models import count_params
from src.train import load_trained, predict_logits

NEURAL = ["mlp", "cnn"]


class SanityError(RuntimeError):
    pass


def wilson(errors: int, shots: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion; well-behaved at 0 and n errors."""
    if shots == 0:
        return 0.0, 1.0
    phat = errors / shots
    denom = 1 + z * z / shots
    centre = (phat + z * z / (2 * shots)) / denom
    half = z * math.sqrt(phat * (1 - phat) / shots + z * z / (4 * shots * shots)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def per_round(ler: float, r: int) -> float:
    return 1 - (1 - ler) ** (1 / r)


def _row(d: int, p: float, decoder: str, wrong: np.ndarray) -> dict:
    k, n = int(wrong.sum()), len(wrong)
    lo, hi = wilson(k, n)
    ler = k / n
    return dict(distance=d, p=p, decoder=decoder, shots=n, errors=k, ler=ler,
                ci_low=lo, ci_high=hi, ler_per_round=per_round(ler, config.rounds(d)))


def nn_predict(model: torch.nn.Module, det: np.ndarray) -> np.ndarray:
    return predict_logits(model, det) > 0


# --------------------------------------------------------------------------- main LER sweep
def ler_sweep(quick: bool, log=print) -> tuple[pd.DataFrame, dict]:
    """LER for every decoder, distance and p. Also returns predictions at FIXED_P for later analyses."""
    mode = config.get_mode(quick)
    rows, keep = [], {}
    for d in config.DISTANCES:
        models = {k: load_trained(k, d) for k in NEURAL}
        fixed_matcher = mwpm.build_matcher(make_circuit(d, config.FIXED_P))
        for p in config.P_SWEEP:
            det, obs = fixed_set("test", d, p, mode["test_shots"][d])
            preds = {
                "pymatching": mwpm.decode(mwpm.build_matcher(make_circuit(d, p)), det),
                "pymatching_fixed_p": mwpm.decode(fixed_matcher, det),
                "no_decoder": np.zeros_like(obs),
            }
            for k, m in models.items():
                preds[k] = nn_predict(m, det)
            for name, pr in preds.items():
                rows.append(_row(d, p, name, pr != obs))
            if p == config.FIXED_P:
                keep[d] = (obs, preds)
            log(f"  d={d} p={p:.3f}  " + "  ".join(f"{n}={(pr != obs).mean():.5f}" for n, pr in preds.items()))
    return pd.DataFrame(rows), keep


# --------------------------------------------------------------------------- analyses
def disagreement(keep: dict) -> pd.DataFrame:
    """At FIXED_P: when a neural decoder and PyMatching disagree, who is right?"""
    rows = []
    for d, (obs, preds) in keep.items():
        mw_ok = preds["pymatching"] == obs
        for k in NEURAL:
            nn_ok = preds[k] == obs
            dis = preds[k] != preds["pymatching"]
            rows.append(dict(distance=d, p=config.FIXED_P, decoder=k, shots=len(obs),
                             disagree=int(dis.sum()), disagree_frac=float(dis.mean()),
                             nn_right=int((dis & nn_ok).sum()), mwpm_right=int((dis & mw_ok).sum()),
                             both_wrong=int((~nn_ok & ~mw_ok).sum())))
    return pd.DataFrame(rows)


def generalisation(quick: bool) -> pd.DataFrame:
    """d=3: mixed-p MLP vs MLP trained only at FIXED_P vs PyMatching, across P_SWEEP."""
    d = 3
    mode = config.get_mode(quick)
    mixed, fixed = load_trained("mlp", d), load_trained("mlp", d, config.FIXED_P)
    rows = []
    for p in config.P_SWEEP:
        det, obs = fixed_set("test", d, p, mode["test_shots"][d])
        for name, pred in [("mlp_mixed_p", nn_predict(mixed, det)),
                           ("mlp_fixed_p", nn_predict(fixed, det)),
                           ("pymatching", mwpm.decode(mwpm.build_matcher(make_circuit(d, p)), det))]:
            rows.append(_row(d, p, name, pred != obs))
    return pd.DataFrame(rows)


def threshold(ler: pd.DataFrame) -> pd.DataFrame:
    """Crossing of PyMatching d=3 and d=5 curves via linear interpolation of log(LER5) - log(LER3) in p."""
    pm = ler[ler.decoder == "pymatching"].pivot(index="p", columns="distance", values="ler").sort_index()
    ps = pm.index.to_numpy()
    diff = np.log(pm[5].clip(lower=1e-12).to_numpy()) - np.log(pm[3].clip(lower=1e-12).to_numpy())
    for i in range(len(ps) - 1):
        if diff[i] < 0 <= diff[i + 1]:
            p_star = ps[i] + (ps[i + 1] - ps[i]) * (-diff[i]) / (diff[i + 1] - diff[i])
            return pd.DataFrame([dict(decoder="pymatching", crossing_found=True, p_threshold=p_star,
                                      p_below=ps[i], p_above=ps[i + 1])])
    return pd.DataFrame([dict(decoder="pymatching", crossing_found=False, p_threshold=float("nan"),
                              p_below=float("nan"), p_above=float("nan"))])


def _median_time(fn, repeats: int = config.TIMING_REPEATS) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def throughput(quick: bool) -> pd.DataFrame:
    """Per-shot decode time on this CPU at FIXED_P: big-batch throughput and single-shot latency."""
    mode = config.get_mode(quick)
    n_big, n_single = mode["throughput_shots"], 200
    torch.set_num_threads(os.cpu_count() or 1)
    rows = []
    for d in config.DISTANCES:
        det, _ = fixed_set("test", d, config.FIXED_P, mode["test_shots"][d])
        big, singles = det[:n_big], det[:n_single]
        matcher = mwpm.build_matcher(make_circuit(d, config.FIXED_P))
        cases = [("pymatching", n_big, len(big), lambda: mwpm.decode(matcher, big)),
                 ("pymatching", 1, n_single, lambda: [matcher.decode(s.astype(np.uint8)) for s in singles])]
        for k in NEURAL:
            m = load_trained(k, d)
            xs = torch.from_numpy(singles.astype(np.float32))
            bx = torch.from_numpy(det[:4096 * max(1, n_big // 4096)].astype(np.float32))

            def batched(m=m, bx=bx):
                with torch.no_grad():
                    for i in range(0, len(bx), 4096):
                        m(bx[i:i + 4096])

            def single(m=m, xs=xs):
                with torch.no_grad():
                    for i in range(len(xs)):
                        m(xs[i:i + 1])

            cases += [(k, 4096, len(bx), batched), (k, 1, n_single, single)]
        for name, bs, n, fn in cases:
            sec = _median_time(fn) / n
            rows.append(dict(distance=d, p=config.FIXED_P, decoder=name, batch_size=bs, shots_timed=n,
                             us_per_shot=sec * 1e6, shots_per_s=1 / sec))
    return pd.DataFrame(rows)


def model_sizes() -> pd.DataFrame:
    rows = []
    for d in config.DISTANCES:
        for k in NEURAL:
            rows.append(dict(model=k, distance=d, params=count_params(load_trained(k, d))))
    rows.append(dict(model="mlp_fixed_p", distance=3, params=count_params(load_trained("mlp", 3, config.FIXED_P))))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- sanity checks
def sanity_checks(ler: pd.DataFrame, log=print) -> list[str]:
    """Raise SanityError (after printing SANITY FAIL) if any physics/statistics check fails."""
    fails = []
    pm = ler[ler.decoder == "pymatching"].set_index(["distance", "p"])
    for p in [p for p in config.P_SWEEP if p <= 0.004]:
        if not pm.loc[(5, p), "ler"] < pm.loc[(3, p), "ler"]:
            fails.append(f"PyMatching d=5 LER not below d=3 at p={p}")
    for d in config.DISTANCES:
        sub = pm.loc[d].sort_index()
        for (p1, a), (p2, b) in zip(sub.iterrows(), list(sub.iterrows())[1:]):
            if b.ci_high < a.ci_low:  # decrease beyond statistical noise
                fails.append(f"PyMatching d={d} LER decreases from p={p1} to p={p2}")
    pmax = max(config.P_SWEEP)
    by = ler.set_index(["distance", "p", "decoder"])
    for d in config.DISTANCES:
        for k in NEURAL:
            if not by.loc[(d, pmax, k), "ler"] < by.loc[(d, pmax, "no_decoder"), "ler"]:
                fails.append(f"{k} d={d} does not beat no_decoder at p={pmax}")
    if not (ler.errors <= ler.shots).all():
        fails.append("errors > shots somewhere")
    if not ((ler.ci_low <= ler.ler) & (ler.ler <= ler.ci_high)).all():
        fails.append("a CI does not contain its point estimate")
    if fails:
        for f in fails:
            log(f"SANITY FAIL: {f}")
        raise SanityError("; ".join(fails))
    log("SANITY OK: all checks passed")
    return fails


# --------------------------------------------------------------------------- entry point
def hardware_line() -> str:
    cpu = platform.processor() or platform.machine()
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return f"{cpu}; {os.cpu_count()} logical CPUs; torch threads {torch.get_num_threads()}; CPU only"


def run(quick: bool, log=print) -> None:
    config.RESULTS.mkdir(exist_ok=True)
    log("evaluating LER on fixed test sets ...")
    ler, keep = ler_sweep(quick, log)
    ler.to_csv(config.RESULTS / "ler_results.csv", index=False)
    sanity_checks(ler, log)
    threshold(ler).to_csv(config.RESULTS / "threshold.csv", index=False)
    disagreement(keep).to_csv(config.RESULTS / "disagreement.csv", index=False)
    log("generalisation (fixed-p vs mixed-p MLP) ...")
    generalisation(quick).to_csv(config.RESULTS / "generalisation.csv", index=False)
    model_sizes().to_csv(config.RESULTS / "model_sizes.csv", index=False)
    log("timing ...")
    tp = throughput(quick)
    with open(config.RESULTS / "throughput.csv", "w") as f:
        f.write(f"# hardware: {hardware_line()}\n")
        tp.to_csv(f, index=False)
    meta = {"mode": "quick" if quick else "full", "test_shots": config.get_mode(quick)["test_shots"],
            "hardware": hardware_line()}
    (config.RESULTS / "eval_meta.json").write_text(json.dumps(meta, indent=2))
    log("evaluation done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true")
    run(ap.parse_args().quick)
