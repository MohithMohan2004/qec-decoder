"""Sampling from Stim, cached val/test sets, an on-the-fly training stream and the CNN grid map."""
from __future__ import annotations

import numpy as np
import stim

import config
from src.circuits import make_circuit

GridIndex = tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]


def sample(circuit: stim.Circuit, shots: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (detection_events [shots, n_det] bool, observable_flips [shots] bool)."""
    sampler = circuit.compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots, separate_observables=True)
    return det, obs[:, 0]


# --------------------------------------------------------------------------- fixed val/test sets
def fixed_set(split: str, d: int, p: float, shots: int) -> tuple[np.ndarray, np.ndarray]:
    """Fixed validation/test set for one (d, p), sampled once and cached as bit-packed .npz."""
    assert split in ("val", "test")
    config.CACHE.mkdir(parents=True, exist_ok=True)
    path = config.CACHE / f"{split}_d{d}_p{p:.4f}_n{shots}.npz"
    circuit = make_circuit(d, p)
    n_det = circuit.num_detectors
    if path.exists():
        z = np.load(path)
        det = np.unpackbits(z["det"], axis=1, count=n_det).astype(bool)
        return det, z["obs"]
    det, obs = sample(circuit, shots, config.split_seed(split, d, p))
    np.savez_compressed(path, det=np.packbits(det, axis=1), obs=obs)
    return det, obs


def mixed_val_set(d: int, total_shots: int, ps: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate equal-sized validation sets over several p values."""
    ps = ps or config.P_SWEEP
    per_p = total_shots // len(ps)
    parts = [fixed_set("val", d, p, per_p) for p in ps]
    return np.concatenate([a for a, _ in parts]), np.concatenate([b for _, b in parts])


# --------------------------------------------------------------------------- training stream
class TrainStream:
    """Fresh training batches from Stim. One precompiled sampler per p; p chosen per sub-batch."""

    def __init__(self, d: int, ps: list[float], seed: int, sub_batches: int = config.SUB_BATCHES):
        self.ps = list(ps)
        self.rng = np.random.default_rng(seed)
        self.sub_batches = sub_batches
        self.samplers = [
            make_circuit(d, p).compile_detector_sampler(seed=seed + 7 * i) for i, p in enumerate(self.ps)
        ]

    def next_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        sub = batch_size // self.sub_batches
        dets, obss = [], []
        for k in self.rng.integers(len(self.ps), size=self.sub_batches):
            det, obs = self.samplers[k].sample(sub, separate_observables=True)
            dets.append(det)
            obss.append(obs[:, 0])
        return np.concatenate(dets), np.concatenate(obss)


# --------------------------------------------------------------------------- grid map for the CNN
def build_grid_index(circuit: stim.Circuit) -> GridIndex:
    """Map each detector to (t, y, x) integer cells by rank-encoding its Stim coordinates."""
    coords = circuit.get_detector_coordinates()
    idx = sorted(coords)
    xs = sorted({coords[i][0] for i in idx})
    ys = sorted({coords[i][1] for i in idx})
    ts = sorted({coords[i][2] for i in idx})
    xr = {v: k for k, v in enumerate(xs)}
    yr = {v: k for k, v in enumerate(ys)}
    tr = {v: k for k, v in enumerate(ts)}
    ti = np.array([tr[coords[i][2]] for i in idx])
    yi = np.array([yr[coords[i][1]] for i in idx])
    xi = np.array([xr[coords[i][0]] for i in idx])
    cells = set(zip(ti.tolist(), yi.tolist(), xi.tolist()))
    assert len(cells) == len(idx), "two detectors collide in the grid map"
    return ti, yi, xi, (len(ts), len(ys), len(xs))


def to_grid(det: np.ndarray, index: GridIndex) -> np.ndarray:
    """[shots, n_det] -> [shots, T, H, W] float32 grid (zeros where no detector lives)."""
    ti, yi, xi, shape = index
    g = np.zeros((det.shape[0], *shape), dtype=np.float32)
    g[:, ti, yi, xi] = det
    return g


def flat_grid_index(index: GridIndex) -> np.ndarray:
    """Flattened cell index of each detector in a [T*H*W] vector (used inside the CNN)."""
    ti, yi, xi, (_, h, w) = index
    return ti * h * w + yi * w + xi
