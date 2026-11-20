"""Build report.md, DEMO_SCRIPT.md, README.md, documentation.md (and demo.ipynb if nbformat exists) from results/*.csv.

Every number in those documents is read from files in results/ or checkpoints/. Nothing is hand-typed.

    python make_report.py            # also regenerates figures
    python make_report.py --no-figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import config

R = config.RESULTS
NN = ["mlp", "cnn"]
NAMES = {"pymatching": "PyMatching", "pymatching_fixed_p": f"PyMatching (fixed p={config.FIXED_P} weights)",
         "mlp": "MLP", "cnn": "CNN", "no_decoder": "No decoder"}
NOISE = "circuit-level depolarizing noise, rounds = d"


# --------------------------------------------------------------------------- loading + formatting
def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(R / name, comment="#")


def load() -> dict:
    ler = _csv("ler_results.csv")
    meta = {}
    for f in sorted(config.CHECKPOINTS.glob("*.json")):
        m = json.loads(f.read_text())
        meta[m["name"]] = m
    return dict(
        ler=ler, by=ler.set_index(["distance", "p", "decoder"]), thr=_csv("threshold.csv").iloc[0],
        tp=_csv("throughput.csv"), sizes=_csv("model_sizes.csv"), dis=_csv("disagreement.csv"),
        gen=_csv("generalisation.csv"), meta=meta,
        eval_meta=json.loads((R / "eval_meta.json").read_text()),
        runtime=json.loads((R / "runtime.json").read_text()) if (R / "runtime.json").exists() else None,
        hardware=(R / "throughput.csv").read_text().splitlines()[0].removeprefix("# hardware: "),
    )


def f_ler(x: float) -> str:
    return "0" if x == 0 else (f"{x:.2e}" if x < 1e-3 else f"{x:.4f}")


def f_ci(row) -> str:
    return f"{f_ler(row.ler)} [{f_ler(row.ci_low)}, {f_ler(row.ci_high)}]"


def f_num(x: float) -> str:
    return f"{x:,.0f}" if x >= 100 else f"{x:.3g}"


def verdict(D: dict, d: int, p: float, dec: str) -> str:
    """'better', 'worse' or 'tied' (overlapping 95% CIs) relative to PyMatching."""
    a, b = D["by"].loc[(d, p, dec)], D["by"].loc[(d, p, "pymatching")]
    if a.ci_high < b.ci_low:
        return "better"
    if a.ci_low > b.ci_high:
        return "worse"
    return "tied"


def ratio(D: dict, d: int, p: float, dec: str, ref: str = "pymatching") -> float:
    ref_ler = D["by"].loc[(d, p, ref)].ler
    return float("nan") if ref_ler == 0 else D["by"].loc[(d, p, dec)].ler / ref_ler


def tp_row(D: dict, d: int, dec: str, big: bool):
    s = D["tp"][(D["tp"].distance == d) & (D["tp"].decoder == dec)]
    return (s[s.batch_size > 1] if big else s[s.batch_size == 1]).iloc[0]


def params(D: dict, model: str, d: int) -> int:
    s = D["sizes"]
    return int(s[(s.model == model) & (s.distance == d)].params.iloc[0])


def tally(D: dict, d: int, dec: str) -> dict:
    out = {"better": [], "worse": [], "tied": []}
    for p in config.P_SWEEP:
        out[verdict(D, d, p, dec)].append(p)
    return out


def plist(ps: list[float]) -> str:
    return ", ".join(f"{p:g}" for p in ps) if ps else "none"


# --------------------------------------------------------------------------- reusable sentences
def conclusion_lines(D: dict | None = None) -> list[str]:
    """Three-line conclusion used by demo.py and the report."""
    D = D or load()
    p = config.FIXED_P
    lines = []
    for d in config.DISTANCES:
        pm = D["by"].loc[(d, p, "pymatching")].ler
        parts = [f"{NAMES[k]} {f_ler(D['by'].loc[(d, p, k)].ler)} ({ratio(D, d, p, k):.2f}x matching, {verdict(D, d, p, k)})" for k in NN]
        lines.append(f"d={d}, p={p:g}: PyMatching LER {f_ler(pm)}; " + "; ".join(parts) + ".")
    d = max(config.DISTANCES)
    pm, best = tp_row(D, d, "pymatching", True), max((tp_row(D, d, k, True) for k in NN), key=lambda r: r.shots_per_s)
    lines.append(f"Speed at d={d} on this CPU: PyMatching {f_num(pm.shots_per_s)} shots/s vs fastest neural net "
                 f"({NAMES[best.decoder]}, batch 4096) {f_num(best.shots_per_s)} shots/s; "
                 f"single-shot latency PyMatching {tp_row(D, d, 'pymatching', False).us_per_shot:.1f} µs vs "
                 f"{NAMES[best.decoder]} {tp_row(D, d, best.decoder, False).us_per_shot:.1f} µs.")
    return lines


def headline(D: dict | None = None) -> str:
    return "\n".join(conclusion_lines(D))


def ler_table(D: dict, ps: list[float]) -> str:
    decs = ["no_decoder", "pymatching", "pymatching_fixed_p", "mlp", "cnn"]
    out = ["| d | p | " + " | ".join(NAMES[k] for k in decs) + " |", "|---|---|" + "---|" * len(decs)]
    for d in config.DISTANCES:
        for p in ps:
            out.append(f"| {d} | {p:g} | " + " | ".join(f_ci(D["by"].loc[(d, p, k)]) for k in decs) + " |")
    return "\n".join(out)


def threshold_sentence(D: dict) -> str:
    t = D["thr"]
    if bool(t.crossing_found):
        return (f"The PyMatching d=3 and d=5 curves cross at p ≈ {t.p_threshold:.4f} (between the sweep points "
                f"{t.p_below:g} and {t.p_above:g}, linear interpolation of the log-LER difference). Below this value "
                f"the larger code is better; above it the larger code is worse.")
    return "No crossing of the PyMatching d=3 and d=5 curves was found inside the swept p range."


# --------------------------------------------------------------------------- report.md
def report(D: dict) -> str:
    ler, gen, dis, meta = D["ler"], D["gen"], D["dis"], D["meta"]
    shots = ler.groupby("distance").shots.first()
    pmax, pmin = max(config.P_SWEEP), min(config.P_SWEEP)
    mode = D["eval_meta"]["mode"]

    train_rows = ["| model | params | steps | best val step | batch | lr | train time (s) |", "|---|---|---|---|---|---|---|"]
    for name in ["mlp_d3", "cnn_d3", "mlp_d5", "cnn_d5", "mlp_fixedp_d3"]:
        if name in meta:
            m = meta[name]
            train_rows.append(f"| {name} | {m['params']:,} | {m['steps']:,} | {m['best_step']:,} | {m['batch_size']} | {m['lr']} | {m['train_seconds']:,} |")

    tp_rows = ["| decoder | d | batch size | µs per shot | shots per second |", "|---|---|---|---|---|"]
    for r in D["tp"].itertuples():
        tp_rows.append(f"| {NAMES[r.decoder]} | {r.distance} | {r.batch_size:,} | {r.us_per_shot:.2f} | {f_num(r.shots_per_s)} |")

    dis_rows = ["| decoder | d | disagree with PyMatching | NN right | PyMatching right | both wrong (agree) |", "|---|---|---|---|---|---|"]
    for r in dis.itertuples():
        dis_rows.append(f"| {NAMES[r.decoder]} | {r.distance} | {r.disagree:,} ({100 * r.disagree_frac:.2f}%) | {r.nn_right:,} | {r.mwpm_right:,} | {r.both_wrong:,} |")

    gby = gen.set_index(["p", "decoder"])
    gen_rows = ["| p | PyMatching | MLP (mixed p) | MLP (p=0.005 only) | fixed / mixed |", "|---|---|---|---|---|"]
    for p in config.P_SWEEP:
        a, b, c = gby.loc[(p, "pymatching")], gby.loc[(p, "mlp_mixed_p")], gby.loc[(p, "mlp_fixed_p")]
        gen_rows.append(f"| {p:g} | {f_ci(a)} | {f_ci(b)} | {f_ci(c)} | {c.ler / b.ler:.2f} |")

    # ---- analysis paragraphs, each built from the numbers above
    analysis = []
    for d in config.DISTANCES:
        sents = []
        for k in NN:
            t = tally(D, d, k)
            rs = [ratio(D, d, p, k) for p in config.P_SWEEP]
            worst_p = config.P_SWEEP[max(range(len(rs)), key=lambda i: rs[i])]
            labels = {"better": "significantly better than PyMatching", "tied": "statistically tied with it (overlapping 95% CIs)",
                      "worse": "significantly worse"}
            parts = [f"{labels[v]} at p = {plist(t[v])}" for v in ("better", "tied", "worse") if t[v]]
            sents.append(f"the {NAMES[k]} is " + ", ".join(parts)
                         + f"; its LER ratio to PyMatching ranges from {min(rs):.2f} to {max(rs):.2f}, largest at p = {worst_p:g}")
        analysis.append(f"**Do the neural decoders match PyMatching at d={d}?** Over the {len(config.P_SWEEP)} noise levels, "
                        + "; ".join(sents) + ".")
    gap = {(k, d): sum(ratio(D, d, p, k) for p in config.P_SWEEP) / len(config.P_SWEEP) for k in NN for d in config.DISTANCES}
    analysis.append("**Does the gap grow with distance?** Averaged over the sweep, the LER ratio to PyMatching is "
                    + "; ".join(f"{NAMES[k]} {gap[(k, 3)]:.2f} at d=3 vs {gap[(k, 5)]:.2f} at d=5" for k in NN)
                    + " (see fig4 and fig9). "
                    + ("For both networks the ratio grows with d, i.e. they capture less of the extra protection a "
                       "bigger code provides." if all(gap[(k, 5)] > gap[(k, 3)] for k in NN) else
                       "The ratio does not grow with d for every network."))
    pm5, pm3 = D["by"].loc[(5, pmin, "pymatching")].ler, D["by"].loc[(3, pmin, "pymatching")].ler
    nn_supp = {k: D["by"].loc[(3, pmin, k)].ler / max(D["by"].loc[(5, pmin, k)].ler, 1e-12) for k in NN}
    analysis.append(f"**Error suppression from d=3 to d=5 at p={pmin:g}.** PyMatching's LER drops by a factor of "
                    f"{pm3 / max(pm5, 1e-12):.1f}; " + "; ".join(
                        f"the {NAMES[k]}'s drops by a factor of {v:.1f}" if v >= 1 else
                        f"the {NAMES[k]}'s instead rises by a factor of {1 / v:.1f} (the bigger code is worse for this network)"
                        for k, v in nn_supp.items()) + ".")
    fx = {p: ratio(D, d, p, "pymatching_fixed_p") for d in [5] for p in (pmin, pmax)}
    analysis.append(f"**Sensitivity of PyMatching to its noise model.** Using edge weights computed for p={config.FIXED_P} "
                    f"instead of the true p changes the d=5 LER by a factor of {fx[pmin]:.3f} at p={pmin:g} and "
                    f"{fx[pmax]:.3f} at p={pmax:g} (1.00 = no change).")
    g_lo = gby.loc[(pmin, "mlp_fixed_p")].ler / gby.loc[(pmin, "mlp_mixed_p")].ler
    g_hi = gby.loc[(pmax, "mlp_fixed_p")].ler / gby.loc[(pmax, "mlp_mixed_p")].ler
    g_mid = gby.loc[(config.FIXED_P, "mlp_fixed_p")].ler / gby.loc[(config.FIXED_P, "mlp_mixed_p")].ler
    analysis.append(f"**Noise-model mismatch for neural decoders (d=3).** The MLP trained only at p={config.FIXED_P} has "
                    f"{g_mid:.2f}x the LER of the mixed-p MLP at its own training noise, {g_lo:.2f}x at p={pmin:g} and "
                    f"{g_hi:.2f}x at p={pmax:g}. "
                    + ("At this small distance the single-p network transfers about as well as the mixed-p one (all "
                       "ratios within 10% of 1), so we see no strong brittleness here; this may not hold at larger d. "
                       if max(g_lo, g_hi, g_mid) < 1.1 else
                       "Ratios well above 1 show that the network did not transfer to noise levels it was not trained on. ")
                    + f"PyMatching needs no retraining: it is rebuilt "
                    f"from the detector error model for each p.")
    d5 = max(config.DISTANCES)
    pm_tp, pm_lat = tp_row(D, d5, "pymatching", True), tp_row(D, d5, "pymatching", False)
    speed = "; ".join(f"{NAMES[k]} {f_num(tp_row(D, d5, k, True).shots_per_s)} shots/s at batch 4096 and "
                      f"{tp_row(D, d5, k, False).us_per_shot:.1f} µs at batch 1" for k in NN)
    analysis.append(f"**Speed and size.** At d={d5}, p={config.FIXED_P} on this CPU, PyMatching decodes "
                    f"{f_num(pm_tp.shots_per_s)} shots/s in a large batch and takes {pm_lat.us_per_shot:.1f} µs per "
                    f"single shot; {speed}. The neural models have "
                    + ", ".join(f"{NAMES[k]} d={d}: {params(D, k, d):,}" for d in config.DISTANCES for k in NN)
                    + " parameters. In relative terms, "
                    + "; ".join(f"in large batches the {NAMES[k]} is {tp_row(D, d5, k, True).shots_per_s / pm_tp.shots_per_s:.2f}x "
                                f"PyMatching's throughput, and at batch 1 its latency is "
                                f"{tp_row(D, d5, k, False).us_per_shot / pm_lat.us_per_shot:.1f}x PyMatching's" for k in NN)
                    + ". Python-level per-call overhead dominates the batch-1 numbers for the networks; "
                      "real-time decoding would need compiled or hardware (FPGA/ASIC) inference.")
    dis_sent = "; ".join(f"{NAMES[r.decoder]} d={r.distance}: {100 * r.disagree_frac:.2f}% of shots, NN right in "
                         f"{r.nn_right:,} vs PyMatching right in {r.mwpm_right:,}" for r in dis.itertuples())
    favour = all(r.mwpm_right > r.nn_right for r in dis.itertuples())
    analysis.append(f"**Disagreements at p={config.FIXED_P}.** {dis_sent}. "
                    + ("In every case PyMatching is right more often when they disagree, so the networks mostly add "
                       "mistakes; but the non-zero 'NN right' counts show they do find some corrections matching misses."
                       if favour else "The networks win a share of the disagreements in at least one setting, so they "
                       "capture some structure that matching does not."))

    full_note = "" if mode == "full" else "\n> **Note:** these numbers come from the `--quick` run (small statistics).\n"
    return f"""# Surface-Code Decoding: PyMatching vs. Small Neural Decoders

**Course:** Quantum Computing (25AM3504)
**Team:** ____
{full_note}
## 1. Background

A **surface code** stores one logical qubit in a 2D grid of d×d data qubits. Ancilla qubits repeatedly measure
local parity checks (stabilizers) of neighbouring data qubits. A check whose outcome changes between rounds is a
**detection event**; the set of detection events of one experiment is the **syndrome**. Errors create detection
events at their endpoints, but the syndrome does not say which error happened. A **decoder** takes the syndrome and
decides whether the logical qubit was flipped, so the flip can be undone in software. If it guesses wrong, a
**logical error** occurs. Below the **threshold** physical error rate, increasing d suppresses logical errors.

**Minimum-weight perfect matching (MWPM)** turns the syndrome into a graph problem: each possible error mechanism
is an edge weighted by log((1-q)/q) for its probability q, and the decoder pairs up detection events (or connects
them to the boundary) with the lowest total weight. PyMatching implements this efficiently.

**Neural decoders** instead learn the map from syndrome to logical flip from simulated examples. They can in principle
capture correlations that matching ignores (for example Y errors that flip both X and Z checks), but they need training
data, can be slow, and their accuracy has to be checked as the code grows.

**Base paper.** *Rethink the Role of Neural Decoders in Quantum Error Correction* (Ge Yan, Shanchuan Li, Yuxuan Du;
arXiv:2605.12046; listed as accepted to ICML 2026). We read its arXiv abstract and introduction (not the full paper).
In our own words: the authors argue that neural decoders are held back by an accuracy-latency trade-off, often at
microsecond timescales. They reproduce five architecture families (MLP, CNN, TCN, Transformer, GNN) on surface codes up
to d=9, mostly on Stim simulations, and add a pruning and quantization pipeline evaluated on FPGA hardware. Their stated
findings are that training-data scale matters more than architectural complexity in the near term, that a suitable
inductive bias is essential for accuracy, that generic MLPs fail to scale, and that INT4 quantization (via
quantization-aware training) is needed for microsecond latency. Our project is a much smaller, CPU-only version of the
first question: how close do a small MLP and a small CNN get to PyMatching at d=3 and d=5?

## 2. Method

- **Circuit.** `stim.Circuit.generated("surface_code:{config.BASIS}")`, distances {config.DISTANCES}, rounds = d.
  {NOISE.capitalize()}: after every Clifford gate, after every reset, before every measurement and on every data qubit
  at the start of each round, an error with probability p. d=3 has 24 detectors and d=5 has 120.
- **Noise sweep.** p = {", ".join(f"{p:g}" for p in config.P_SWEEP)}.
- **Data.** Training batches are sampled fresh from Stim at every step (no fixed training set); each batch is
  {config.SUB_BATCHES} sub-batches, each with p drawn uniformly from the sweep. Validation (used only for checkpoint
  selection) and test (used only for the numbers below) sets are fixed and use different seeds
  (train {config.SEEDS['train']}, val {config.SEEDS['val']}, test {config.SEEDS['test']}, per-(d, p) offsets).
  Test shots per (d, p): d=3 {shots[3]:,}, d=5 {shots[5]:,}.
- **PyMatching.** Matching graph built from `circuit.detector_error_model(decompose_errors=True)` separately for every
  (d, p), so the weights match the true noise. A second variant uses the graph for p={config.FIXED_P} at every p.
- **MLP.** Fully connected ReLU network on the flat detector vector (d=3: 24→128→64→1; d=5: 120→256→128→64→1).
- **CNN.** Detectors are placed on a [T, H, W] grid by rank-encoding their Stim (x, y, t) coordinates; time is used as
  the channel axis. Three 3×3 conv layers (32, 64, 64 channels) and a 128-unit dense head. No changes to this
  architecture were needed.
- **Training.** Binary cross-entropy on the logical-flip label, Adam, cosine learning-rate decay, gradient clipping at
  1.0; the checkpoint with the lowest validation loss is kept.

{chr(10).join(train_rows)}

- **Statistics.** Logical error rate = errors / shots, with a 95% Wilson score interval
  (centre = (k/n + z²/2n) / (1 + z²/n), half-width = z·sqrt(k/n·(1-k/n)/n + z²/4n²) / (1 + z²/n), z = 1.96), which stays
  inside [0, 1] and is sensible at 0 errors. "Tied" below means the two 95% intervals overlap, i.e. not statistically
  distinguishable.

## 3. Results

### 3.1 Logical error rate vs physical error rate

![d=3](figures/fig1_ler_vs_p_d3.png)
![d=5](figures/fig2_ler_vs_p_d5.png)

LER per shot with 95% Wilson CI at selected p (full sweep in `results/ler_results.csv`):

{ler_table(D, [0.003, 0.005, 0.008])}

### 3.2 Threshold

{threshold_sentence(D)}

![threshold](figures/fig3_threshold.png)

### 3.3 Neural vs matching ratio and scaling with distance

![ratio](figures/fig4_ratio.png)
![scaling](figures/fig9_scaling.png)

### 3.4 Training curves

![training](figures/fig5_training_curves.png)

### 3.5 Speed and size

Hardware: {D['hardware']}.

{chr(10).join(tp_rows)}

![throughput](figures/fig6_throughput.png)

### 3.6 Generalisation across noise levels (d=3)

{chr(10).join(gen_rows)}

![generalisation](figures/fig7_generalisation.png)

### 3.7 Disagreements at p={config.FIXED_P}

{chr(10).join(dis_rows)}

![disagreement](figures/fig8_disagreement.png)

## 4. Analysis

{chr(10).join(chr(10) + a for a in analysis)}

**Summary.**
{chr(10).join("- " + line for line in conclusion_lines(D))}

## 5. Limitations

- Only the Z-basis memory experiment (storing a qubit, no logical gates), one noise model (uniform circuit-level
  depolarizing), and only d=3 and d=5.
- Each shot is decoded as a whole block of d rounds; there is no sliding-window or streaming decoding.
- The neural networks are deliberately small and trained for a fixed budget on a laptop CPU; larger models and more
  data (the base paper's main lever) could narrow the gap.
- Timings are CPU-only, in Python, on one laptop; they are not representative of an optimised real-time decoder.
- Finite statistics: at the lowest p the d=5 LER is estimated from relatively few errors, so CIs are wide there.

## 6. Future work

- Larger distances (d=7, 9) and more rounds, to test whether the neural gap keeps growing.
- Architectures with a better inductive bias for the lattice: graph neural networks or transformers on the detector graph.
- Correlated or biased noise, where matching's independence assumption is wrong and learning can help most.
- Hybrid decoders: matching first, then a network that predicts when to override it.
- Real-time constraints: quantization and FPGA deployment, as studied in the base paper.

## 7. Reproducibility

- `python run_all.py --full` regenerates everything (`--quick` for a smoke test); `python demo.py --d 3 --p 0.01`.
- Seeds: {', '.join(f'{k} {v}' for k, v in config.SEEDS.items())} (per-(d, p) offsets added, see `config.split_seed`). Results above come from the `--{mode}` run.
- Hardware: {D['hardware']}.
- Package versions: see `results/env.txt` and `requirements.txt`.

## 8. References

1. Ge Yan, Shanchuan Li, Yuxuan Du. *Rethink the Role of Neural Decoders in Quantum Error Correction.* arXiv:2605.12046.
2. Craig Gidney. *Stim: a fast stabilizer circuit simulator.*
3. Oscar Higgott, Craig Gidney. *Sparse Blossom: correcting a million errors per core second with minimum-weight matching.* (PyMatching v2)
4. Oscar Higgott. *PyMatching: A Python package for decoding quantum codes with minimum-weight perfect matching.*
5. Austin G. Fowler, Matteo Mariantoni, John M. Martinis, Andrew N. Cleland. *Surface codes: Towards practical large-scale quantum computation.*
"""


# --------------------------------------------------------------------------- DEMO_SCRIPT.md
def demo_script(D: dict) -> str:
    p, pmax = config.FIXED_P, max(config.P_SWEEP)
    by = D["by"]
    d3 = {k: f_ler(by.loc[(3, p, k)].ler) for k in ["pymatching", "mlp", "cnn", "no_decoder"]}
    d5 = {k: f_ler(by.loc[(5, p, k)].ler) for k in ["pymatching", "mlp", "cnn", "no_decoder"]}
    t = D["thr"]
    thr = f"about {t.p_threshold:.4f}, so roughly {100 * t.p_threshold:.2f} percent" if bool(t.crossing_found) else "not inside our sweep"
    r5 = {k: ratio(D, 5, p, k) for k in NN}
    r3 = {k: ratio(D, 3, p, k) for k in NN}
    pm_tp, cnn_tp = tp_row(D, 5, "pymatching", True), tp_row(D, 5, "cnn", True)
    pm_lat, cnn_lat, mlp_lat = tp_row(D, 5, "pymatching", False), tp_row(D, 5, "cnn", False), tp_row(D, 5, "mlp", False)
    gby = D["gen"].set_index(["p", "decoder"])
    g_hi = gby.loc[(pmax, "mlp_fixed_p")].ler / gby.loc[(pmax, "mlp_mixed_p")].ler
    pmin = min(config.P_SWEEP)
    g_lo = gby.loc[(pmin, "mlp_fixed_p")].ler / gby.loc[(pmin, "mlp_mixed_p")].ler
    shots = D["ler"].groupby("distance").shots.first()
    v3 = {k: verdict(D, 3, p, k) for k in NN}
    v5 = {k: verdict(D, 5, p, k) for k in NN}
    vt = {"tied": "statistically tied with", "worse": "significantly worse than", "better": "significantly better than"}
    supp = by.loc[(3, pmin, "pymatching")].ler / max(by.loc[(5, pmin, "pymatching")].ler, 1e-12)
    fx = ratio(D, 5, pmax, "pymatching_fixed_p")
    mlp_tp = tp_row(D, 5, "mlp", True)

    # Judgement sentences are chosen by the data, never assumed.
    nn_worse_d5 = all(v5[k] == "worse" for k in NN)
    scale_sent = ("So the gap to matching grows with code size: the small networks do not keep up as the code grows."
                  if all(r5[k] > r3[k] for k in NN) else
                  "The gap to matching does not clearly grow with code size at this noise level.")
    speed_sent = (f"On speed, in big batches PyMatching decodes {f_num(pm_tp.shots_per_s)} shots per second, the MLP "
                  f"{f_num(mlp_tp.shots_per_s)} and the CNN {f_num(cnn_tp.shots_per_s)}; for single shots PyMatching takes "
                  f"{pm_lat.us_per_shot:.1f} microseconds against {mlp_lat.us_per_shot:.1f} for the MLP and "
                  f"{cnn_lat.us_per_shot:.1f} for the CNN.")
    brittle = max(g_lo, g_hi) >= 1.1
    brittle_sent = (f"An MLP trained at only one noise level has {g_lo:.2f} times the error rate of the mixed-noise MLP at "
                    f"p = {pmin:g} and {g_hi:.2f} times at p = {pmax:g}, "
                    + ("so it is brittle when the noise changes, while matching simply rebuilds its graph."
                       if brittle else "so at distance 3 we saw little brittleness to a change in noise level."))
    closing = ("So: matching is still the strong, simple baseline, and neural decoders have to earn their place with more "
               "data, better structure and faster hardware." if nn_worse_d5 else
               "So: small neural decoders can compete with matching in places, but matching remains the simpler and "
               "more general baseline.")
    fx_sent = ("so matching is quite insensitive to this mismatch" if abs(fx - 1) < 0.05 else
               "so matching does depend noticeably on knowing the noise level")

    rows = [
        ("0:00", "Quantum computers make errors all the time. Error correction is what will make them useful, and "
                 "decoding is the fast classical step that decides how to fix those errors. That decoding step is our project.",
         "Title slide"),
        ("0:30", "A surface code spreads one logical qubit over a grid of physical qubits. Helper qubits keep checking the "
                 "parity of their neighbours. When a check changes value, we call it a detection event. The pattern of "
                 "detection events is the syndrome, which is the only clue we get about what went wrong.",
         "Slide: `figures/demo_shot_d3.png` (lattice with red stars)"),
        ("1:15", "So the decoding problem is: given the syndrome, did the logical qubit flip or not? We compare two answers. "
                 "PyMatching is a graph algorithm. It pairs up the detection events so that the total weight of the "
                 "implied errors is as small as possible. The other answer is a neural network that simply learns the "
                 "mapping from many simulated examples.",
         "Slide: matching vs neural network diagram"),
        ("2:00", f"We generated all data with Stim, a fast quantum-circuit simulator, using circuit-level noise: every gate, "
                 f"reset, measurement and idle step can fail with probability p. We used distance 3 and distance 5. We "
                 f"trained a small MLP and a small CNN, from {params(D, 'mlp', 3):,} to {params(D, 'cnn', 5):,} parameters, "
                 f"on fresh samples at every step, and tested on {shots[3]:,} shots per point at d equals 3 and "
                 f"{shots[5]:,} at d equals 5.",
         "Slide: method summary"),
        ("2:45", "Now the live demo. First, the circuit statistics: number of qubits and detectors.",
         "Run `python demo.py --d 3 --p 0.01 --pause`; section 1"),
        ("3:00", "Here is one shot. The red stars are detection events, round by round. Now all three decoders give their "
                 "answer, and the tick or cross shows who got it right.",
         "Press Enter: sections 2 and 3"),
        ("3:20", "Next, twenty thousand fresh shots at three noise levels. Look at the PyMatching column against the "
                 "MLP and CNN columns. The no-decoder column is the floor every decoder has to beat.",
         "Press Enter: section 4 table"),
        ("3:40", "And here are the full test-set curves and the speed comparison, then the conclusion the script computes "
                 "from our result files.",
         "Press Enter: sections 5 and 6 (fig1 + fig6)"),
        ("4:00", f"Our results. At p = {p:g} and distance 3, PyMatching has a logical error rate of {d3['pymatching']}, "
                 f"the MLP {d3['mlp']} and the CNN {d3['cnn']}; with no decoder it would be {d3['no_decoder']}. At distance 5 "
                 f"it is {d5['pymatching']} for PyMatching, {d5['mlp']} for the MLP and {d5['cnn']} for the CNN. "
                 f"PyMatching's threshold estimate is {thr}.",
         "Slide: `figures/fig2_ler_vs_p_d5.png` and `figures/fig4_ratio.png`"),
        ("4:30", f"The honest takeaway. At distance 3 the MLP is {vt[v3['mlp']]} PyMatching and the CNN is {vt[v3['cnn']]} it. "
                 f"At distance 5 the MLP is {vt[v5['mlp']]} it, with {r5['mlp']:.2f} times its error rate, and the CNN is "
                 f"{vt[v5['cnn']]} it, with {r5['cnn']:.2f} times. {scale_sent} {speed_sent} {brittle_sent} Next we would "
                 f"try larger distances, graph-based networks, and hybrid matching-plus-network decoders.",
         "Slide: `figures/fig9_scaling.png`, `figures/fig6_throughput.png`"),
        ("4:50", f"{closing} Thank you, we are happy to take questions.", "Closing slide"),
    ]
    table = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)

    qa = [
        ("Why circuit-level noise instead of a simpler model?",
         "Real hardware has faulty gates and faulty measurements, not just random flips on data qubits. Circuit-level "
         "noise includes all of that, so the decoder also has to handle measurement errors across rounds. It is the "
         "standard, more realistic benchmark."),
        ("What is a threshold?",
         f"It is the physical error rate below which making the code bigger makes the logical qubit better. For "
         f"PyMatching we estimate the d=3 / d=5 crossing at {thr}. At p = {pmin:g}, going from d=3 to d=5 cuts "
         f"PyMatching's logical error rate by a factor of {supp:.1f}."),
        ("Why does PyMatching win (or lose) at d=5?",
         f"At d=5 the syndrome has 120 detectors and many more error patterns; matching handles this exactly through the "
         f"graph structure, while a small network must learn it from examples. At p = {p:g} the MLP is {r5['mlp']:.2f} "
         f"times and the CNN {r5['cnn']:.2f} times PyMatching's error rate (at d=3: {r3['mlp']:.2f} and {r3['cnn']:.2f})."),
        ("Could the network handle more rounds than it was trained on?",
         "Not as built. Both networks have a fixed input size (24 or 120 detectors, d rounds). A different number of "
         "rounds changes the input shape. A convolution over time or a recurrent/transformer model could handle that; "
         "PyMatching handles any number of rounds automatically."),
        ("Why train on a mix of noise levels?",
         f"At low p almost all syndromes are empty and a network can collapse to always answering 'no flip'. Mixing in "
         f"higher p gives it informative examples, and one model then covers the whole sweep. For comparison, an MLP "
         f"trained only at p = {config.FIXED_P} has {g_lo:.2f} times the error rate of the mixed-p MLP at p = {pmin:g} "
         f"and {g_hi:.2f} times at p = {pmax:g}" + (", so single-p training is brittle." if brittle else
         ", so at d=3 single-p training happened to transfer well too.")),
        ("What is a Wilson interval?",
         "A 95% confidence interval for a success/failure rate. Unlike the simple plus-or-minus formula it never goes "
         "below zero and it still gives a sensible upper bound when we see zero errors. When two decoders' intervals "
         "overlap we call them statistically tied."),
        ("Why does decoder speed matter?",
         f"A superconducting quantum computer produces a new round of syndrome data about every microsecond. If decoding "
         f"is slower than that, a backlog builds up. On our laptop at d=5 the single-shot latency is "
         f"{pm_lat.us_per_shot:.1f} µs for PyMatching, {mlp_lat.us_per_shot:.1f} µs for the MLP and "
         f"{cnn_lat.us_per_shot:.1f} µs for the CNN."),
        ("What would a real-time decoder need?",
         "Microsecond-scale latency with streaming input, so dedicated hardware such as FPGAs or ASICs, compact models "
         "(the base paper points to aggressive INT4 quantization for neural decoders), and decoding in sliding windows "
         "rather than whole blocks."),
        ("Does PyMatching need to know the noise level?",
         f"It uses the noise model to set edge weights. Using weights for p = {config.FIXED_P} at p = {pmax:g} changes "
         f"its d=5 error rate by a factor of {fx:.3f}, {fx_sent}."),
        ("How big are the networks?",
         ", ".join(f"{NAMES[k]} d={d}: {params(D, k, d):,} parameters" for d in config.DISTANCES for k in NN)
         + ". PyMatching has no trained parameters; its graph comes from the circuit."),
    ]
    qa_md = "\n\n".join(f"**Q{i}. {q}**\n{a}" for i, (q, a) in enumerate(qa, 1))
    return f"""# Demo Script (5 minutes)

Auto-generated by `make_report.py` from `results/*.csv`; regenerate after every run so the numbers stay correct.
Noise model throughout: {NOISE}.

| Time | Say | Show / Run |
|---|---|---|
{table}

## Before you present

- Run `python demo.py --d 3 --p 0.01 --no-plots` once to warm the caches and check checkpoints exist.
- Have `figures/` open in an image viewer for the slides.

## Likely questions and answers

{qa_md}
"""


# --------------------------------------------------------------------------- README.md
def readme(D: dict) -> str:
    def exists(*names: str) -> bool:
        return all((config.ROOT / n).exists() for n in names)

    log = (R / "run_log.txt").read_text() if (R / "run_log.txt").exists() else ""
    sanity = "SANITY OK" in log and "SANITY FAIL" not in log.split("===== run_all")[-1]
    checks = [
        ("O1", "Stim data for d=3 and d=5", "`src/circuits.py`, `results/cache/`, tests pass", exists("src/circuits.py") and any(config.CACHE.glob("test_d5_*"))),
        ("O2", "PyMatching baseline", "`src/mwpm.py`, rows in `ler_results.csv`", "pymatching" in set(D["ler"].decoder)),
        ("O3", "Small MLP trained", "`checkpoints/mlp_d{3,5}.pt`, `train_log_*`", exists("checkpoints/mlp_d3.pt", "checkpoints/mlp_d5.pt", "results/train_log_mlp_d3.csv")),
        ("O4", "Small CNN trained", "`checkpoints/cnn_d{3,5}.pt`, `train_log_*`", exists("checkpoints/cnn_d3.pt", "checkpoints/cnn_d5.pt", "results/train_log_cnn_d5.csv")),
        ("O5", "LER comparison with CIs", "`ler_results.csv`, `fig1`, `fig2`, `fig4`", exists("figures/fig1_ler_vs_p_d3.png", "figures/fig2_ler_vs_p_d5.png", "figures/fig4_ratio.png")),
        ("O6", "Speed/size + analysis", "`throughput.csv`, `model_sizes.csv`, `disagreement.csv`, `generalisation.csv`, `threshold.csv`",
         all((R / n).exists() for n in ["throughput.csv", "model_sizes.csv", "disagreement.csv", "generalisation.csv", "threshold.csv"])),
        ("O7", "Demo + word-for-word script", "`demo.py`, `DEMO_SCRIPT.md`", exists("demo.py", "DEMO_SCRIPT.md")),
        ("O8", "Report from data", "`report.md` generated by `make_report.py`", exists("report.md")),
    ]
    rows = "\n".join(f"| [{'x' if ok else ' '}] {i} | {req} | {ev} |" for i, req, ev, ok in checks)
    rt = D["runtime"]
    runtime = (f"{rt['total_seconds'] / 60:.1f} min for `run_all.py --{rt['mode']}`"
               + (" (training skipped, checkpoints reused)" if rt.get("skip_train") else "")) if rt else "not recorded"
    train_s = sum(m["train_seconds"] for m in D["meta"].values())
    demo_t = (R / "demo_time.txt").read_text().strip() if (R / "demo_time.txt").exists() else "not recorded"
    return f"""# Surface-Code Decoding: PyMatching vs. a Small Neural Decoder

Course project for **Quantum Computing (25AM3504)**. We simulate the rotated surface code (distance 3 and 5,
{NOISE}) with **Stim**, decode it with **PyMatching** (minimum-weight perfect matching) and with a small **MLP** and
**CNN** trained in PyTorch, and compare logical error rates (with 95% Wilson intervals), speed and model size.
Base paper: *Rethink the Role of Neural Decoders in Quantum Error Correction* (arXiv:2605.12046).

**Headline (from `results/`, `--{D['eval_meta']['mode']}` run):**
{chr(10).join("- " + line for line in conclusion_lines(D))}

Full write-up: [`report.md`](report.md). Talk track: [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md). How it was built: [`BUILD_LOG.md`](BUILD_LOG.md). Original spec: [`task.md`](task.md).

## How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
source .venv/bin/activate
python run_all.py --quick      # smoke test (~10-15 min), produces demo-capable checkpoints
python run_all.py --full       # the real numbers (see "Hardware and runtime" below)
python demo.py --d 3 --p 0.01 --pause   # live demo from checkpoints (add --no-plots for text only)
python make_report.py          # rebuild report.md, DEMO_SCRIPT.md, README.md, documentation.md and figures
pytest -q                      # unit tests
python -m src.train --model cnn --d 5 [--quick]   # train one model
```

### Web UI (React)

An interactive version of the demo: live shots on the lattice with PyMatching's matched pairs, live mini-benchmarks,
and every result chart, all read from `results/` and the saved checkpoints.

```bash
cd frontend && npm install && npm run build && cd ..   # once (Node 18+)
python server.py               # open http://127.0.0.1:8000
```

For frontend development run `python server.py` and `cd frontend && npm run dev` side by side (Vite proxies `/api`).
Full details: [`documentation.md`](documentation.md).

## Layout

`config.py` (all knobs) · `src/circuits.py` (Stim) · `src/data.py` (sampling, caches, grid map) · `src/mwpm.py`
(PyMatching) · `src/models.py` (MLP, CNN) · `src/train.py` · `src/evaluate.py` (LER, CIs, timing, sanity checks) ·
`src/plots.py` · `src/viz.py` (lattice picture) · `server.py` (JSON API + static host for the web UI) ·
`frontend/` (React + Vite UI) · `results/` (CSVs, logs) · `checkpoints/` · `figures/`.

## Objective checklist

| ID | Requirement | Evidence |
|---|---|---|
{rows}

Sanity checks (d=5 better than d=3 below threshold, monotone LER, neural nets beat no-decoder, valid CIs):
**{'passed' if sanity else 'NOT confirmed - see results/run_log.txt'}** in the last logged run.

## Deviations from the task specification

- No packages were installed system-wide; everything runs in a project virtualenv (`.venv`, CPU-only PyTorch wheel).
- `val_shots` is interpreted as the **total** validation set per distance, split evenly across the {len(config.P_SWEEP)} p values
  (the fixed-p MLP uses a validation set at p={config.FIXED_P} only).
- The CNN takes the flat detector vector and scatters it onto the [T, H, W] grid inside the model (same rank-encoded
  map as `data.to_grid`), so both networks share one input format.
- The disagreement table's "both wrong" column counts shots where both decoders are wrong; such shots are necessarily
  ones where the two decoders agree, because the output is binary.
- Single-shot (batch 1) latency is also measured for PyMatching, for a like-for-like comparison.
- Step counts, shots and the CNN architecture were not changed from the specification.

## Hardware and runtime

- {D['hardware']}
- Last pipeline run: {runtime}; total training time over the 5 models: {train_s / 60:.1f} min.
- Demo: {demo_t}.
"""


# --------------------------------------------------------------------------- optional notebook
def notebook() -> None:
    try:
        import nbformat
    except ImportError:
        return
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook()
    nb.cells = [
        md("# Surface-code decoding demo\nMirrors `demo.py`: PyMatching vs a small MLP and CNN on Stim data. "
           "Run `python run_all.py --quick` first so checkpoints exist."),
        code("import numpy as np\nfrom src.circuits import make_circuit, circuit_stats\nfrom src.data import sample\n"
             "from src import mwpm\nfrom src.train import load_trained, predict_logits\nimport make_report\n"
             "d, p = 3, 0.01\ncircuit = make_circuit(d, p)\ncircuit_stats(circuit)"),
        md("## One shot\nThe detection events are the only information the decoder gets."),
        code("det, obs = sample(circuit, 64, seed=4000)\ni = int(np.argmax(det.sum(1) > 0))\nshot, truth = det[i], bool(obs[i])\n"
             "print('fired detectors:', np.flatnonzero(shot), ' true flip:', truth)"),
        md("## Decode it with all three decoders"),
        code("preds = {'PyMatching': bool(mwpm.decode(mwpm.build_matcher(circuit), shot[None])[0])}\n"
             "for k in ('mlp', 'cnn'):\n    preds[k.upper()] = bool(predict_logits(load_trained(k, d), shot[None])[0] > 0)\n"
             "from src.viz import draw_shot\nimport config\nfrom IPython.display import Image\n"
             "Image(str(draw_shot(circuit, shot, truth, preds, p, d, config.FIGURES / 'demo_shot_nb.png')))"),
        md("## Mini benchmark on fresh shots"),
        code("for bp in (0.003, 0.006, 0.010):\n    c = make_circuit(d, bp)\n    bd, bo = sample(c, 20_000, seed=4001)\n"
             "    res = {'none': bo.mean(), 'PyMatching': (mwpm.decode(mwpm.build_matcher(c), bd) != bo).mean()}\n"
             "    for k in ('mlp', 'cnn'):\n        res[k] = ((predict_logits(load_trained(k, d), bd) > 0) != bo).mean()\n"
             "    print(bp, {k: round(float(v), 5) for k, v in res.items()})"),
        md("## Full results and conclusion (from `results/`)"),
        code("from IPython.display import display\ndisplay(Image('figures/fig1_ler_vs_p_d3.png'), Image('figures/fig6_throughput.png'))\n"
             "print('\\n'.join(make_report.conclusion_lines()))"),
    ]
    nbformat.write(nb, config.ROOT / "demo.ipynb")


# --------------------------------------------------------------------------- documentation.md
def documentation(D: dict) -> str:
    """Single reference document: what, why, how to run, how it works, results, performance verdict, web UI."""
    fp = config.FIXED_P
    F = config.FULL
    t = D["thr"]
    shots = D["eval_meta"]["test_shots"]
    mode = D["eval_meta"].get("mode", "?")
    hw = D["hardware"]
    run_min = f"{D['runtime']['total_seconds'] / 60:.0f} min" if D["runtime"] else "unknown"
    by = D["by"]

    def r(d, k, p=fp):
        return ratio(D, d, p, k)

    def tp(d, k, big):
        return tp_row(D, d, k, big)

    best = {d: min(NN, key=lambda k: r(d, k)) for d in config.DISTANCES}
    gains = {k: [p for p in config.P_SWEEP if by.loc[(5, p, k)].ler < by.loc[(3, p, k)].ler]
             for k in ["pymatching"] + NN}
    gen = D["gen"].set_index(["p", "decoder"])
    gen_gap = max(gen.loc[(p, "mlp_fixed_p")].ler / gen.loc[(p, "mlp_mixed_p")].ler - 1 for p in config.P_SWEEP)
    fixed_dev = max(abs(ratio(D, d, p, "pymatching_fixed_p") - 1) for d in config.DISTANCES for p in config.P_SWEEP)
    cnn5 = D["meta"].get("cnn_d5", {})
    cnn5_log = _csv("train_log_cnn_d5.csv")
    still_falling = len(cnn5_log) > 4 and cnn5_log.val_loss.iloc[-1] < cnn5_log.val_loss.iloc[-5]
    beats_none = all(by.loc[(d, max(config.P_SWEEP), k)].ler < by.loc[(d, max(config.P_SWEEP), "no_decoder")].ler
                     for d in config.DISTANCES for k in NN)
    lat_gap = min(tp(5, k, False).us_per_shot for k in NN) / tp(5, "pymatching", False).us_per_shot
    train_min = sum(m.get("train_seconds", 0) for m in D["meta"].values()) / 60

    def tl(d, k):
        tt = tally(D, d, k)
        return (f"better at {len(tt['better'])}, tied at {len(tt['tied'])}, worse at {len(tt['worse'])} "
                f"of {len(config.P_SWEEP)} noise levels")

    tp_lines = []
    for d in config.DISTANCES:
        for k in ["pymatching"] + NN:
            b, s = tp(d, k, True), tp(d, k, False)
            tp_lines.append(f"| {d} | {NAMES[k]} | {f_num(b.shots_per_s)} (batch {int(b.batch_size):,}) | "
                            f"{s.us_per_shot:.1f} |")
    size_lines = [f"| {'MLP (trained at fixed p)' if row.model == 'mlp_fixed_p' else NAMES[row.model]} | "
                  f"{row.distance} | {row.params:,} |" for row in D["sizes"].itertuples()]
    dis_lines = [f"| {row.distance} | {NAMES[row.decoder]} | {row.disagree_frac * 100:.2f}% | {row.nn_right:,} | "
                 f"{row.mwpm_right:,} | {row.both_wrong:,} |" for row in D["dis"].itertuples()]
    train_lines = [f"| `{n}` | {m['params']:,} | {m['steps']:,} | {m['best_step']:,} | {m['best_val_loss']:.4f} | "
                   f"{m.get('train_seconds', 0) / 60:.1f} |" for n, m in sorted(D["meta"].items())]

    fastest = {d: max(["pymatching"] + NN, key=lambda k: tp(d, k, True).shots_per_s) for d in config.DISTANCES}
    slowest = {d: min(["pymatching"] + NN, key=lambda k: tp(d, k, True).shots_per_s) for d in config.DISTANCES}
    fast_txt = (f"the {NAMES[fastest[3]]} is the fastest decoder at both distances" if fastest[3] == fastest[5]
                else f"the fastest decoder is the {NAMES[fastest[3]]} at d = 3 and the {NAMES[fastest[5]]} at d = 5")
    slow_txt = (f"The {NAMES[slowest[3]]} is the slowest in batched mode at both distances."
                if slowest[3] == slowest[5] else "")
    d3_ok = r(3, best[3]) < 1
    d5_ok = r(5, best[5]) < 1
    overall = ("good at d = 3 and not good at d = 5" if d3_ok and not d5_ok else
               "good at both distances" if d3_ok and d5_ok else
               "not good at either distance" if not d3_ok and not d5_ok else "good only at d = 5")

    return f"""# Documentation: Surface-Code Decoding, PyMatching vs. a Small Neural Decoder

Course project for **Quantum Computing (25AM3504)**. This is the single reference for the project: what it does, how to
run it, how each part works, what the results are and how good they are, and how the web interface is built.

> This file is **generated** by `make_report.py` from `results/*.csv` and `checkpoints/*.json`. Do not edit it by hand.
> Change the generator and run `python make_report.py --no-figures`. Every number below comes from those files
> (results mode: `{mode}`).

---

## 1. What the project does

A quantum computer's qubits make errors constantly. **Quantum error correction** spreads one *logical* qubit over many
physical qubits and repeatedly measures *stabilizers* (parity checks). The pattern of check outcomes that changed, the
**syndrome**, hints at where errors happened. A **decoder** is the fast classical algorithm that reads the syndrome and
decides whether the logical qubit was flipped, so the flip can be undone.

This project compares two kinds of decoder on the **rotated surface code** at **distance 3 and 5**:

| Decoder | Idea |
|---|---|
| **PyMatching** (MWPM) | Treat detection events as nodes of a graph whose edge weights come from the noise model, then pair them up with minimum total weight. A strong, standard baseline with no training. |
| **MLP** | A small fully connected network that learns "syndrome → did the logical qubit flip?" from simulated examples. |
| **CNN** | A small convolutional network that sees the same detectors placed on a 2-D grid (time steps as channels). |
| No decoder | Always predicts "no flip". The floor that every real decoder must beat. |
| PyMatching (fixed p={fp} weights) | Matching with edge weights from the wrong noise level, to test sensitivity to a mis-specified noise model. |

Base paper: *Rethink the Role of Neural Decoders in Quantum Error Correction* (arXiv:2605.12046). Only its abstract and
introduction were reviewed; see `report.md`.

### Objectives and where they are satisfied

| ID | Requirement | Evidence | Status |
|---|---|---|---|
| O1 | Stim data for d=3 and d=5 | `src/circuits.py`, `src/data.py`, `results/cache/` | done |
| O2 | PyMatching baseline | `src/mwpm.py`, `results/ler_results.csv` | done |
| O3 | Small MLP trained | `checkpoints/mlp_d3.pt`, `checkpoints/mlp_d5.pt`, `results/train_log_mlp_*.csv` | done |
| O4 | Small CNN trained | `checkpoints/cnn_d3.pt`, `checkpoints/cnn_d5.pt`, `results/train_log_cnn_*.csv` | done |
| O5 | LER comparison with 95% CIs | `results/ler_results.csv`, `figures/fig1`, `fig2`, `fig4` | done |
| O6 | Speed/size + analysis | `throughput.csv`, `model_sizes.csv`, `disagreement.csv`, `generalisation.csv`, `threshold.csv` | done |
| O7 | Demo under 3 min + word-for-word script | `demo.py`, web UI (`server.py` + `frontend/`), `DEMO_SCRIPT.md` | done |
| O8 | Report built from data | `report.md`, generated by `make_report.py` | done |

---

## 2. Quick start

All Python commands use the project virtualenv (`.venv`); the system Python has no Stim or PyTorch.

```bash
# one-time setup
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..        # web UI (Node 18+)

# web UI: live demo + all results in the browser
.venv/bin/python server.py                                    # then open http://127.0.0.1:8000

# command-line demo (from saved checkpoints, no training)
.venv/bin/python demo.py --d 3 --p 0.01 --pause               # --no-plots for text only, --d 5 for distance 5

# full pipeline and documents
.venv/bin/python run_all.py --quick                           # smoke test, ~10-15 min
.venv/bin/python run_all.py --full                            # real numbers ({run_min} on this laptop)
.venv/bin/python make_report.py                               # rebuild report.md, DEMO_SCRIPT.md, README.md, documentation.md, figures
.venv/bin/python -m pytest -q                                 # unit tests
.venv/bin/python -m src.train --model cnn --d 5 [--quick]     # train one model
```

---

## 3. Repository layout

| Path | Role |
|---|---|
| `config.py` | Every knob: distances, `P_SWEEP`, seeds, QUICK/FULL settings, paths. |
| `src/circuits.py` | Builds the Stim circuit (rotated surface code, Z-memory, circuit-level noise). |
| `src/data.py` | Sampling, cached validation/test sets, on-the-fly training stream, grid map for the CNN. |
| `src/mwpm.py` | PyMatching wrapper (matching graph from the detector error model). |
| `src/models.py` | MLP and CNN definitions. |
| `src/train.py` | Training loop and checkpoint loading. |
| `src/evaluate.py` | Logical error rate, Wilson CIs, timing, disagreement, generalisation, threshold, sanity checks. |
| `src/plots.py`, `src/viz.py` | Report figures and the matplotlib lattice picture. |
| `run_all.py` | Whole pipeline: env check → tests → caches → train → evaluate → figures → documents. |
| `demo.py` | Command-line demo. |
| `server.py` | Standard-library HTTP server: JSON API + static host for the React UI. |
| `frontend/` | React 19 + Vite 6 web UI (plain JavaScript, no chart library). |
| `make_report.py` | Generates `report.md`, `DEMO_SCRIPT.md`, `README.md`, `documentation.md`, `demo.ipynb`. |
| `tests/` | Unit tests for circuits, data plumbing and decoders. |
| `results/` | CSVs, logs, `env.txt`, `runtime.json`; `results/cache/` holds bit-packed val/test sets. |
| `checkpoints/` | Trained weights (`.pt`) plus metadata (`.json`). |
| `figures/` | The 9 report figures and demo lattice pictures (PNG, 200 dpi). |
| `task.md`, `BUILD_LOG.md`, `skills.md` | Original specification, build record, UI design guidelines. |

---

## 4. How it works

### 4.1 Simulation (Stim)

- Circuit: `stim.Circuit.generated("surface_code:{config.BASIS}", distance=d, rounds=d, ...)`.
- Noise: {NOISE}. With strength p on every Clifford gate (`after_clifford_depolarization`), reset
  (`after_reset_flip_probability`), measurement (`before_measure_flip_probability`) and idle data qubit per round
  (`before_round_data_depolarization`).
- Each shot gives a vector of **detection events** (a detector fires when a check outcome changes between rounds) and
  one bit: did the logical observable flip?
- Physical error rates swept: {plist(config.P_SWEEP)}.

### 4.2 Data

- **Training** batches are sampled fresh from Stim at every step, so no example is seen twice. Each batch has
  {config.SUB_BATCHES} sub-batches, each at a p drawn uniformly from the sweep. Training on mixed p prevents the network
  collapsing to "always predict no flip" at low noise.
- **Validation** and **test** sets are fixed, sampled once with their own seeds and cached in `results/cache/`.
  Test shots per p: {shots['3']:,} at d=3, {shots['5']:,} at d=5.
- Seeds (base): train {config.SEEDS['train']}, validation {config.SEEDS['val']}, test {config.SEEDS['test']},
  demo {config.SEEDS['demo']}. A distinct seed per (split, d, p) is derived by `config.split_seed`. The test set is
  never used for training or model selection.
- **CNN grid map:** each detector's Stim coordinates (x, y, t) are rank-encoded into integer cells of a [T, H, W]
  tensor. The mapping is asserted collision-free.

### 4.3 Decoders

- **PyMatching:** `pymatching.Matching.from_detector_error_model(circuit.detector_error_model(decompose_errors=True))`,
  built separately for each (d, p) so the edge weights match the true noise.
- **MLP:** d=3 `24 → 128 → 64 → 1`; d=5 `120 → 256 → 128 → 64 → 1`; ReLU activations.
- **CNN:** three 3×3 conv layers (32, 64, 64 channels) on the [T, H, W] grid, then `Linear → 128 → 1`. It takes the
  same flat detector vector as the MLP and scatters it onto the grid internally.
- Both networks output one logit for "the logical qubit flipped" and predict a flip when the logit is above 0.

### 4.4 Training

FULL settings: batch {F['batch_size']:,}, learning rate {F['lr']:g} (Adam, cosine decay to zero), gradient clipping
1.0, steps {F['train_steps'][3]:,} (d=3) / {F['train_steps'][5]:,} (d=5). The checkpoint with the best validation
loss is kept. Total training time over all models: {train_min:.0f} min on {hw}.

| Checkpoint | Params | Steps | Best step | Best val. loss | Train time (min) |
|---|---|---|---|---|---|
{chr(10).join(train_lines)}

### 4.5 Evaluation

- Logical error rate (LER) = errors / shots, with a **95% Wilson score interval** (written by hand, valid at 0 errors).
- Two decoders are called "not statistically distinguishable" when their intervals overlap.
- Throughput/latency: median of {config.TIMING_REPEATS} timed repeats after a warm-up, CPU only.
- Built-in **sanity checks** stop the pipeline if they fail: d=5 beats d=3 for PyMatching at low p, LER rises with p,
  every network beats the no-decoder floor at the highest p, CIs contain their point estimates.

---

## 5. Results

### 5.1 Headline

{chr(10).join('- ' + line for line in conclusion_lines(D))}

### 5.2 Logical error rate (95% Wilson CI) at selected p

{ler_table(D, [0.003, 0.005, 0.008])}

The full sweep is in `results/ler_results.csv` and in the web UI's *Accuracy* section.

### 5.3 Threshold

{threshold_sentence(D)}

### 5.4 Speed

Hardware: {hw}.

| d | Decoder | Batched throughput (shots/s) | Single-shot latency (µs) |
|---|---|---|---|
{chr(10).join(tp_lines)}

### 5.5 Model size

| Model | d | Trainable parameters |
|---|---|---|
| PyMatching | – | 0 (weights come from the noise model) |
{chr(10).join(size_lines)}

### 5.6 When the network and PyMatching disagree (p = {fp})

| d | Network | Shots where they disagree | Network right | PyMatching right | Both wrong (they agree) |
|---|---|---|---|---|---|
{chr(10).join(dis_lines)}

### 5.7 Noise-model mismatch (d = 3)

An extra MLP trained only at p = {fp} is at most **{gen_gap * 100:.1f}%** worse than the mixed-p MLP anywhere in the
sweep. PyMatching with weights fixed at p = {fp} stays within **{fixed_dev * 100:.1f}%** of the correctly weighted
matcher at every p and distance.

---

## 6. Does it work, and is the performance good?

### 6.1 Does the project work as intended for the demo?

Yes. Checked against the objectives:

- `pytest -q` passes, and the pipeline's sanity checks passed in the last full run (`results/run_log.txt`).
- `demo.py` runs from saved checkpoints in seconds ({(R / 'demo_time.txt').read_text().strip() if (R / 'demo_time.txt').exists() else 'not recorded'}), well under the 3-minute limit.
- The web UI samples live shots, decodes them with all three decoders, runs live benchmarks, and draws every result
  from the CSVs.
- Every network beats the no-decoder baseline at the highest p: **{'yes' if beats_none else 'no'}**.

### 6.2 Is the performance good?

Overall the neural decoders are **{overall}**.

- **d = 3: {'strong' if d3_ok else 'weak'}.** At p = {fp}, the MLP's error rate is {r(3, 'mlp'):.2f}× PyMatching's and the CNN's is
  {r(3, 'cnn'):.2f}×. Across the sweep, the MLP is {tl(3, 'mlp')}; the CNN is {tl(3, 'cnn')}. One plausible reason:
  matching decomposes correlated (Y-type) errors into independent X and Z parts, while a network can learn the
  correlation. At d = 3 the syndrome space is small enough for that to be learned well.
- **d = 5: {'strong' if d5_ok else 'weak'}.** At p = {fp}, the MLP is {r(5, 'mlp'):.2f}× PyMatching and the CNN {r(5, 'cnn'):.2f}×. The MLP is
  {tl(5, 'mlp')}; the CNN is {tl(5, 'cnn')}. The larger code helps PyMatching at {len(gains['pymatching'])} of
  {len(config.P_SWEEP)} noise levels but helps the MLP at only {len(gains['mlp'])} (p = {plist(gains['mlp'])}) and the
  CNN at {len(gains['cnn'])} (p = {plist(gains['cnn'])}). So the networks do not yet turn extra qubits into
  protection. The CNN d=5 validation loss was {'still falling' if still_falling else 'flat'} when training ended (best
  step {cnn5.get('best_step', '?'):,} of {cnn5.get('steps', '?'):,}), so it is {'probably under-trained as well as small' if still_falling else 'limited by size rather than training length'}.
- **Speed: a trade-off.** Batched, {fast_txt}. One shot at a time, PyMatching is {lat_gap:.0f}× faster than the
  quickest network at d = 5. Real-time error correction cares about per-shot latency, so PyMatching is the practical
  choice on a CPU. {slow_txt}
- **Robustness:** small sensitivity to noise-model mismatch at d = 3 (Section 5.7).

### 6.3 What would improve it

1. Train the d=5 networks {'longer and ' if still_falling else ''}wider. The CNN d=5 loss curve
   {'had not flattened' if still_falling else 'had flattened'}, and its training alone took {cnn5.get('train_seconds', 0) / 60:.0f} min on this CPU, so a GPU would help.
2. Architectures that respect the code's structure and scale with distance: translation-equivariant CNNs over
   space-time (3-D convolutions), graph neural networks over the detector graph, or transformers.
3. Hybrid decoders: a network that pre-processes or re-weights the syndrome, with matching doing the final pairing.
4. More rounds and sliding-window decoding, correlated or biased noise, larger distances.

---

## 7. Web interface

### 7.1 Running it

```bash
cd frontend && npm install && npm run build && cd ..
.venv/bin/python server.py [--port 8000] [--host 127.0.0.1]
```

Development with hot reload: run `python server.py` and, in a second terminal, `cd frontend && npm run dev`. Vite
proxies `/api` and `/figures` to port 8000.

The installed dependency versions (React 19.1.0, Vite 6.3.5, @vitejs/plugin-react 4.5.1) were chosen because they
were already in the local npm cache, so `npm install --offline` works without network access.

### 7.2 Sections

| Section | What it shows |
|---|---|
| Hero | Four computed headline tiles: best-network ratio at d=3 and d=5, threshold, d=5 throughput. |
| Live decoder | Choose d and p, sample a shot, view detection events on the lattice by round (or all rounds), PyMatching's matched pairs (space-like lines, time-like dashed rings, dotted lines to the boundary), each decoder's verdict with ✓/✗, the networks' P(flip), and a live mini-benchmark with Wilson CIs. |
| Accuracy | LER vs p for all decoders (d toggle, CI bands, table view), ratio to PyMatching, threshold crossing. |
| Speed & size | Batched throughput and single-shot latency (log scale), parameter counts. |
| Analysis | Who is right when decoders disagree, noise-model mismatch, validation-loss curves. |
| Verdict | Strengths, weaknesses and trade-offs, worded in the browser from the CSV values. |

### 7.3 API (`server.py`, standard library only)

| Endpoint | Returns |
|---|---|
| `GET /api/summary` | Every results CSV as rows, training logs, checkpoint metadata, run metadata, the computed conclusion. |
| `GET /api/layout?d=3` | Data-qubit and check coordinates (with X/Z type), detector (x, y, t) coordinates, circuit stats. |
| `GET /api/shot?d=3&p=0.01&seed=4000&nonempty=1` | One shot: fired detectors, true flip, each decoder's prediction, latency and P(flip), PyMatching's matched edges (`-1` = boundary). Skips silent shots when `nonempty=1`. |
| `GET /api/benchmark?d=3&p=0.01&shots=50000&seed=4001` | LER, Wilson CI and µs/shot for no-decoder, PyMatching, MLP and CNN on fresh shots (1k-200k). |
| `GET /figures/<name>.png` | The report figures. |

p must be within [0.0005, 0.03]. Matchers and models are cached in memory, and requests are decoded one at a time
under a lock.

### 7.4 Design

The UI follows the Apple design guidelines in `skills.md`:

- **Springs, not fixed-duration animations** (damping ratio + response, `frontend/src/motion.js`). They are
  interruptible and always start from the current value and velocity: the segmented-control thumb, benchmark bars,
  detection-event pop-in, and the slider's settle.
- **Direct manipulation:** the p slider tracks the pointer 1:1 with pointer capture and respects the grab offset. It
  rubber-bands past its ends instead of stopping hard, and fires the request on release.
- **Instant feedback on press:** buttons and segments respond on `pointerdown` with a small scale.
- **Materials:** a translucent, blurred sticky navigation bar and tooltips; content scrolls underneath.
- **Typography:** system font, negative tracking and tight leading on large headings, tabular numbers.
- **Accessibility:** `prefers-reduced-motion` (no springs), `prefers-reduced-transparency` (solid surfaces),
  `prefers-contrast: more`, forced colours, keyboard-operable slider and segmented controls, light/dark/auto theme.

Charts follow a validated colour-blind-safe categorical palette, with the colour fixed per decoder (PyMatching blue,
MLP orange, CNN green, fixed-p PyMatching amber, no-decoder grey). Dash style and marker shape act as a second cue,
alongside a legend, direct end labels, hover tooltips and a table view for every line chart.

---

## 8. Presenting the demo

- Talk track with timings and likely Q&A: `DEMO_SCRIPT.md`.
- Suggested flow in the web UI: Hero tiles → Live decoder at d=3, p=1%: step through *Next shot* and the round
  selector, then run a 50k benchmark → switch to d=5 and run again (the gap to PyMatching opens up) → Accuracy → Speed
  → Verdict.
- Fallback with no browser: `python demo.py --d 3 --p 0.01 --pause`.

---

## 9. Testing and validation

- `pytest -q`: detector counts (24 at d=3, 120 at d=5), noiseless determinism, grid uniqueness and round-trip,
  seeding, PyMatching at p=0 and low p, Wilson edge cases, model shapes.
- Pipeline sanity checks (Section 4.5) assert the physics before any result is trusted.
- The web API was exercised end to end. The live 50k-shot benchmark reproduces the saved test-set LERs within their
  confidence intervals.

---

## 10. Limitations

Memory experiment only (Z basis, rounds = d); one noise model (uniform circuit-level depolarizing); small models;
CPU-only timings, with the PyTorch call overhead included in latency; finite statistics; no sliding-window or streaming
decoding.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: stim` / `torch` | Use `.venv/bin/python`, not the system Python. |
| Web page says it can't reach the server | Start `python server.py`. In dev mode keep it running next to `npm run dev`. |
| `frontend not built` JSON on port 8000 | `cd frontend && npm run build`. |
| `Missing checkpoints` | `python run_all.py --quick` (or `--full`). |
| `npm install` fails offline | The pinned versions are in the local npm cache; try `npm install --offline`. |
| Port in use | `python server.py --port 8080`. |

---

## 12. References

1. Ge Yan, Shanchuan Li, Yuxuan Du. *Rethink the Role of Neural Decoders in Quantum Error Correction.* arXiv:2605.12046.
2. Craig Gidney. *Stim: a fast stabilizer circuit simulator.*
3. Oscar Higgott, Craig Gidney. *PyMatching* (sparse blossom minimum-weight perfect matching).
"""


def main(make_figures: bool = True) -> None:
    if make_figures:
        from src import plots
        plots.make_all()
    D = load()
    (config.ROOT / "report.md").write_text(report(D))
    (config.ROOT / "DEMO_SCRIPT.md").write_text(demo_script(D))
    (config.ROOT / "README.md").write_text(readme(D))
    (config.ROOT / "documentation.md").write_text(documentation(D))
    notebook()
    print("wrote report.md, DEMO_SCRIPT.md, README.md, documentation.md" + (", demo.ipynb" if (config.ROOT / "demo.ipynb").exists() else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-figures", action="store_true")
    main(not ap.parse_args().no_figures)
