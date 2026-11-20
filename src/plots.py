"""All report figures. Reads results/*.csv, writes figures/*.png (200 dpi)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3, "legend.fontsize": 10})
NOISE = "circuit-level depolarizing, rounds = d"

# Okabe-Ito colour-blind-safe palette; each decoder also gets its own marker + line style.
STYLE = {
    "pymatching":         dict(color="#0072B2", marker="o", ls="-",  label="PyMatching (matched p)"),
    "pymatching_fixed_p": dict(color="#56B4E9", marker="s", ls="--", label=f"PyMatching (fixed p={config.FIXED_P} weights)"),
    "mlp":                dict(color="#E69F00", marker="^", ls="-",  label="MLP"),
    "cnn":                dict(color="#009E73", marker="D", ls="-",  label="CNN"),
    "no_decoder":         dict(color="#999999", marker="x", ls=":",  label="no decoder (always 'no flip')"),
    "mlp_mixed_p":        dict(color="#E69F00", marker="^", ls="-",  label="MLP trained on mixed p"),
    "mlp_fixed_p":        dict(color="#CC79A7", marker="v", ls="--", label=f"MLP trained at p={config.FIXED_P} only"),
}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(config.RESULTS / name, comment="#")


def _save(fig, name: str) -> None:
    config.FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES / name, dpi=200)
    plt.close(fig)


def _ler_axes(ax, df: pd.DataFrame, decoders: list[str]) -> None:
    for dec in decoders:
        s = df[df.decoder == dec].sort_values("p")
        st = STYLE[dec]
        ler = s.ler.clip(lower=1e-7)
        ax.plot(s.p, ler, color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"])
        ax.fill_between(s.p, s.ci_low.clip(lower=1e-7), s.ci_high, color=st["color"], alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p (per operation)")
    ax.set_ylabel("logical error rate (per shot)")
    ax.legend()


def fig_ler(ler: pd.DataFrame) -> None:
    for i, d in enumerate(config.DISTANCES, start=1):
        df = ler[ler.distance == d]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        _ler_axes(ax, df, ["no_decoder", "pymatching_fixed_p", "pymatching", "mlp", "cnn"])
        ax.set_title(f"d={d}: logical error rate vs p (95% Wilson CI bands)\n{NOISE}; {df.shots.iloc[0]:,} test shots per point")
        _save(fig, f"fig{i}_ler_vs_p_d{d}.png")


def fig_threshold(ler: pd.DataFrame, thr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for d, c, m in [(3, "#0072B2", "o"), (5, "#D55E00", "s")]:
        s = ler[(ler.decoder == "pymatching") & (ler.distance == d)].sort_values("p")
        ax.errorbar(s.p, s.ler, yerr=[s.ler - s.ci_low, s.ci_high - s.ler], color=c, marker=m, capsize=3, label=f"PyMatching d={d}")
    t = thr.iloc[0]
    if bool(t.crossing_found):
        ax.axvline(t.p_threshold, color="black", ls="--", label=f"estimated crossing p ≈ {t.p_threshold:.4f}")
    else:
        ax.text(0.05, 0.9, "no crossing inside the sweep", transform=ax.transAxes)
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate p (per operation)")
    ax.set_ylabel("logical error rate (per shot)")
    shots = ler[ler.decoder == "pymatching"].groupby("distance").shots.first()
    ax.set_title(f"PyMatching threshold estimate (d=3 vs d=5 crossing)\n{NOISE}; shots d3={shots[3]:,}, d5={shots[5]:,}")
    ax.legend()
    _save(fig, "fig3_threshold.png")


def fig_ratio(ler: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pm = ler[ler.decoder == "pymatching"].set_index(["distance", "p"]).ler
    for dec in ["mlp", "cnn"]:
        for d, ls in [(3, "-"), (5, "--")]:
            s = ler[(ler.decoder == dec) & (ler.distance == d)].sort_values("p")
            ratio = s.ler.to_numpy() / pm.loc[d].loc[s.p].to_numpy()
            ax.plot(s.p, ratio, color=STYLE[dec]["color"], marker=STYLE[dec]["marker"], ls=ls, label=f"{dec.upper()} d={d}")
    ax.axhline(1.0, color="black", ls="--", lw=1, label="parity with PyMatching")
    ax.set_xlabel("physical error rate p (per operation)")
    ax.set_ylabel("LER(neural) / LER(PyMatching)")
    ax.set_title(f"Neural vs matching LER ratio (<1 = neural better, >1 = neural worse)\n{NOISE}; same test shots as fig1/fig2")
    ax.legend()
    _save(fig, "fig4_ratio.png")


def fig_training() -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for d, ls in [(3, "-"), (5, "--")]:
        for kind in ["mlp", "cnn"]:
            path = config.RESULTS / f"train_log_{kind}_d{d}.csv"
            if path.exists():
                s = pd.read_csv(path)
                ax.plot(s.step, s.val_loss, color=STYLE[kind]["color"], ls=ls, marker=STYLE[kind]["marker"],
                        markevery=5, label=f"{kind.upper()} d={d}")
    ax.set_xlabel("training step (one fresh Stim batch per step)")
    ax.set_ylabel("validation BCE loss (mixed p)")
    ax.set_title(f"Training curves on the fixed validation set\n{NOISE}")
    ax.legend()
    _save(fig, "fig5_training_curves.png")


def fig_throughput(tp: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    order = ["pymatching", "mlp", "cnn"]
    width = 0.38
    for ax, bs_sel, metric, ylabel in [
        (axes[0], "big", "shots_per_s", "throughput (shots / s, large batch)"),
        (axes[1], 1, "us_per_shot", "latency per shot (µs, batch size 1)"),
    ]:
        for j, d in enumerate(config.DISTANCES):
            vals = []
            for dec in order:
                s = tp[(tp.decoder == dec) & (tp.distance == d)]
                s = s[s.batch_size > 1] if bs_sel == "big" else s[s.batch_size == 1]
                vals.append(float(s[metric].iloc[0]))
            x = np.arange(len(order)) + (j - 0.5) * width
            bars = ax.bar(x, vals, width, label=f"d={d}", color=["#0072B2", "#D55E00"][j], hatch=["", "//"][j])
            ax.bar_label(bars, fmt="%.3g", fontsize=9)
        ax.set_xticks(np.arange(len(order)), ["PyMatching", "MLP", "CNN"])
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.legend()
    fig.suptitle(f"Decoder speed on this laptop CPU at p={config.FIXED_P} ({NOISE})\n"
                 f"median of {config.TIMING_REPEATS} repeats; large batch = PyMatching full batch, NNs batch 4096")
    _save(fig, "fig6_throughput.png")


def fig_generalisation(gen: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _ler_axes(ax, gen, ["pymatching", "mlp_mixed_p", "mlp_fixed_p"])
    ax.axvline(config.FIXED_P, color="grey", ls=":", lw=1)
    ax.set_title(f"d=3: does a neural decoder generalise across noise levels?\n{NOISE}; {gen.shots.iloc[0]:,} test shots per point")
    _save(fig, "fig7_generalisation.png")


def fig_disagreement(dis: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6))
    labels = [f"{r.decoder.upper()} d={r.distance}" for r in dis.itertuples()]
    x = np.arange(len(dis))
    nn_right = dis.nn_right.to_numpy()
    mw_right = dis.mwpm_right.to_numpy()
    ax.bar(x, mw_right, color="#0072B2", label="PyMatching right, NN wrong")
    ax.bar(x, nn_right, bottom=mw_right, color="#E69F00", hatch="//", label="NN right, PyMatching wrong")
    for i, r in enumerate(dis.itertuples()):
        ax.text(i, r.disagree * 1.02, f"{100 * r.disagree_frac:.2f}% of shots", ha="center", fontsize=10)
    ax.set_xticks(x, labels)
    ax.set_ylabel("number of disagreeing test shots")
    ax.set_ylim(0, dis.disagree.max() * 1.15)
    ax.set_title(f"When the neural decoder and PyMatching disagree, who is right?\n"
                 f"p={config.FIXED_P}, {NOISE}\nshots d3={dis[dis.distance == 3].shots.iloc[0]:,}, d5={dis[dis.distance == 5].shots.iloc[0]:,}")
    ax.legend()
    _save(fig, "fig8_disagreement.png")


def fig_scaling(ler: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pm = ler[ler.decoder == "pymatching"].set_index(["distance", "p"]).ler
    ps = [p for p in config.P_SWEEP if p <= 0.006]
    x = np.arange(len(config.DISTANCES))
    width = 0.8 / (2 * len(ps))
    k = 0
    for dec in ["mlp", "cnn"]:
        for p in ps:
            vals = [ler[(ler.decoder == dec) & (ler.distance == d) & (ler.p == p)].ler.iloc[0] / pm.loc[(d, p)]
                    for d in config.DISTANCES]
            alpha = 0.35 + 0.65 * (ps.index(p) + 1) / len(ps)
            ax.bar(x - 0.4 + width * (k + 0.5), vals, width, color=STYLE[dec]["color"], alpha=alpha,
                   hatch="" if dec == "mlp" else "..", label=f"{dec.upper()} p={p}")
            k += 1
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set_xticks(x, [f"d={d}" for d in config.DISTANCES])
    ax.set_yscale("log")
    ax.set_ylabel("LER(neural) / LER(PyMatching)")
    ax.set_title(f"Does the neural gap grow with code distance? (p ≤ 0.006)\n{NOISE}; same test shots as fig1/fig2")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, "fig9_scaling.png")


def make_all() -> None:
    ler = _read("ler_results.csv")
    fig_ler(ler)
    fig_threshold(ler, _read("threshold.csv"))
    fig_ratio(ler)
    fig_training()
    fig_throughput(_read("throughput.csv"))
    fig_generalisation(_read("generalisation.csv"))
    fig_disagreement(_read("disagreement.csv"))
    fig_scaling(ler)


if __name__ == "__main__":
    make_all()
