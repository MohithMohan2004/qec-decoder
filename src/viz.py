"""Draw the rotated surface-code patch with one shot's detection events, round by round."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stim
from matplotlib.patches import Polygon

Z_COLOR, X_COLOR, HIT = "#9ecae1", "#fdd0a2", "#d62728"


def _layout(circuit: stim.Circuit):
    """Data-qubit positions, ancilla positions and which ancillas are Z-type."""
    det_coords = circuit.get_detector_coordinates()
    anc = sorted({(c[0], c[1]) for c in det_coords.values()})
    z_type = {(c[0], c[1]) for c in det_coords.values() if c[2] == 0}  # Z-memory: only Z checks fire at t=0
    data = sorted({(x, y) for x, y in circuit.get_final_qubit_coordinates().values() if x % 2 == 1 and y % 2 == 1})
    return data, anc, z_type, det_coords


def _draw_patch(ax, data, anc, z_type):
    dset = set(data)
    for (x, y) in anc:
        nb = [(x + dx, y + dy) for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)] if (x + dx, y + dy) in dset]
        pts = nb if len(nb) == 4 else nb + [(x, y)]
        ax.add_patch(Polygon(pts, closed=True, fc=Z_COLOR if (x, y) in z_type else X_COLOR, ec="grey", lw=0.8))
    ax.scatter(*zip(*data), s=30, c="black", zorder=3)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")


def draw_shot(circuit: stim.Circuit, det_row: np.ndarray, truth: bool, preds: dict[str, bool],
              p: float, d: int, path: Path) -> Path:
    """Small multiples: one panel per detector time layer + a collapsed view. Returns the PNG path."""
    data, anc, z_type, coords = _layout(circuit)
    times = sorted({c[2] for c in coords.values()})
    n = len(times) + 1
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows + 1.4))
    axes = np.atleast_1d(axes).ravel()
    fired = [i for i in range(len(det_row)) if det_row[i]]
    for ax, t in zip(axes, times):
        _draw_patch(ax, data, anc, z_type)
        pts = [(coords[i][0], coords[i][1]) for i in fired if coords[i][2] == t]
        if pts:
            ax.scatter(*zip(*pts), s=180, marker="*", c=HIT, zorder=4)
        label = f"round {int(t) + 1}" if t < len(times) - 1 else "final data readout"
        ax.set_title(f"{label}: {len(pts)} event(s)", fontsize=12)
    ax = axes[len(times)]
    _draw_patch(ax, data, anc, z_type)
    counts: dict[tuple, int] = {}
    for i in fired:
        key = (coords[i][0], coords[i][1])
        counts[key] = counts.get(key, 0) + 1
    for (x, y), c in counts.items():
        ax.scatter([x], [y], s=120 + 80 * c, marker="*", c=HIT, zorder=4)
        ax.text(x + 0.35, y - 0.35, str(c), fontsize=11, color=HIT, zorder=5)
    ax.set_title(f"all rounds: {len(fired)} event(s)", fontsize=12)
    for ax in axes[n:]:
        ax.axis("off")

    verdict = "   ".join(f"{name}: {'flip' if v else 'no flip'} {'✓' if v == truth else '✗'}" for name, v in preds.items())
    fig.suptitle(f"d={d} rotated surface code, one shot at p={p} (circuit-level depolarizing, rounds = d)\n"
                 f"blue = Z checks, orange = X checks, red star = detection event\n"
                 f"TRUE logical flip: {'YES' if truth else 'no'}    |    {verdict}", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
