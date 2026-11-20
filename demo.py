"""Live demo from saved checkpoints (no training). Runs in well under 3 minutes on a laptop CPU.

    python demo.py --d 3 --p 0.01 --pause
    python demo.py --d 5 --no-plots
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib

import config

DEMO_PS = [0.003, 0.006, 0.010]
BENCH_SHOTS = 20_000


def header(n: int, title: str, pause: bool) -> None:
    if pause and n > 1:
        input("\n   [Enter] for the next section ...")
    print(f"\n{'=' * 72}\n {n}. {title}\n{'=' * 72}")


def show_png(paths: list, enabled: bool) -> None:
    """Open PNGs in a window when a GUI backend exists; otherwise just print where they are."""
    for p in paths:
        print(f"   figure: {p}")
    if not enabled or matplotlib.get_backend().lower() == "agg":
        return
    import matplotlib.pyplot as plt
    for p in paths:
        fig = plt.figure(figsize=(10, 7))
        plt.imshow(plt.imread(p))
        plt.axis("off")
        fig.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d", type=int, choices=config.DISTANCES, default=3)
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=config.SEEDS["demo"])
    ap.add_argument("--no-plots", action="store_true", help="print only, do not draw or open figures")
    ap.add_argument("--pause", action="store_true", help="wait for Enter between sections")
    a = ap.parse_args()
    if a.no_plots or not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
        matplotlib.use("Agg")

    missing = [n for n in (f"mlp_d{a.d}.pt", f"cnn_d{a.d}.pt") if not (config.CHECKPOINTS / n).exists()]
    if missing:
        sys.exit(f"Missing checkpoints {missing}. Run `python run_all.py --quick` (or --full) first.")

    import numpy as np
    import torch

    from src import mwpm
    from src.circuits import circuit_stats, make_circuit
    from src.data import sample
    from src.evaluate import wilson
    from src.train import load_trained, predict_logits
    import make_report

    torch.set_num_threads(os.cpu_count() or 1)
    t0 = time.perf_counter()
    d, p = a.d, a.p

    header(1, f"The Stim circuit: rotated surface code, d={d}, {config.rounds(d)} rounds, p={p}", a.pause)
    circuit = make_circuit(d, p)
    for k, v in circuit_stats(circuit).items():
        print(f"   {k:<13}{v}")
    print(f"   noise model  circuit-level depolarizing: every gate, reset, measurement and idle step has error rate {p}")

    header(2, "Sample one shot and look at its syndrome", a.pause)
    det, obs = sample(circuit, 64, seed=a.seed)
    i = int(np.argmax(det.sum(1) > 0))  # first shot with at least one detection event
    shot, truth = det[i], bool(obs[i])
    print(f"   shot #{i} of a seed-{a.seed} batch: {int(shot.sum())} of {len(shot)} detectors fired")
    print(f"   fired detector indices: {np.flatnonzero(shot).tolist()}")
    print(f"   did the logical qubit actually flip?  {'YES' if truth else 'no'}")

    header(3, "Decode it: PyMatching vs MLP vs CNN", a.pause)
    matcher = mwpm.build_matcher(circuit)
    models = {"MLP": load_trained("mlp", d), "CNN": load_trained("cnn", d)}
    preds = {"PyMatching": bool(mwpm.decode(matcher, shot[None])[0])}
    for name, m in models.items():
        preds[name] = bool(predict_logits(m, shot[None])[0] > 0)
    for name, v in preds.items():
        print(f"   {name:<11} predicts {'flip   ' if v else 'no flip'}  {'✓ correct' if v == truth else '✗ wrong'}")
    if not a.no_plots:
        from src.viz import draw_shot
        path = draw_shot(circuit, shot, truth, preds, p, d, config.FIGURES / f"demo_shot_d{d}.png")
        show_png([path], True)

    header(4, f"Mini-benchmark: {BENCH_SHOTS:,} fresh shots per p (demo seed, not the test set)", a.pause)
    print(f"   {'p':>6} | {'no decoder':>11} | {'PyMatching':>11} | {'MLP':>11} | {'CNN':>11}   (logical error rate)")
    print("   " + "-" * 66)
    for j, bp in enumerate(DEMO_PS):
        c = make_circuit(d, bp)
        bd, bo = sample(c, BENCH_SHOTS, seed=a.seed + 1 + j)
        res = [bo.mean(), (mwpm.decode(mwpm.build_matcher(c), bd) != bo).mean()]
        res += [((predict_logits(m, bd) > 0) != bo).mean() for m in models.values()]
        print(f"   {bp:>6.3f} | " + " | ".join(f"{r:>11.5f}" for r in res))
    lo, hi = wilson(int(0.01 * BENCH_SHOTS), BENCH_SHOTS)
    print(f"   (with {BENCH_SHOTS:,} shots an LER of 0.01 has a 95% CI of about [{lo:.4f}, {hi:.4f}])")

    header(5, "Full results (from the saved test-set evaluation)", a.pause)
    fig = config.FIGURES / f"fig{config.DISTANCES.index(d) + 1}_ler_vs_p_d{d}.png"
    show_png([fig, config.FIGURES / "fig6_throughput.png"], not a.no_plots)

    header(6, "Conclusion (computed from results/*.csv)", a.pause)
    for line in make_report.conclusion_lines():
        print(f"   - {line}")
    elapsed = time.perf_counter() - t0
    print(f"\n   demo finished in {elapsed:.1f} s")
    if not a.pause:  # record an unattended run time for the README
        (config.RESULTS / "demo_time.txt").write_text(
            f"last unattended `demo.py --d {d} --p {p}{' --no-plots' if a.no_plots else ''}` took {elapsed:.1f} s")
    if a.pause and not a.no_plots and matplotlib.get_backend().lower() != "agg":
        input("   [Enter] to close the figures ...")


if __name__ == "__main__":
    main()
