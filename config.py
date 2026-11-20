"""Single source of truth for every experiment knob.

All scripts import from here, so changing a value here changes it everywhere.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CACHE = RESULTS / "cache"
CHECKPOINTS = ROOT / "checkpoints"
FIGURES = ROOT / "figures"

# --------------------------------------------------------------------------- physics
DISTANCES = [3, 5]
BASIS = "rotated_memory_z"
P_SWEEP = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012]  # physical error rates
FIXED_P = 0.005           # noise level for the "fixed weights" matcher, fixed-p MLP and timing


def rounds(d: int) -> int:
    """A memory experiment lasts d rounds of stabilizer measurement."""
    return d


# --------------------------------------------------------------------------- seeds
# Train / val / test use disjoint base seeds. Per-(d, p) seeds are derived from these.
SEEDS = {"train": 1000, "val": 2000, "test": 3000, "demo": 4000}


def split_seed(split: str, d: int, p: float) -> int:
    """Deterministic seed for one (split, d, p) combination."""
    return SEEDS[split] + 100 * d + P_SWEEP.index(p) if p in P_SWEEP else SEEDS[split] + 100 * d + 99


# --------------------------------------------------------------------------- run modes
# val_shots is the TOTAL validation set size per distance, split evenly over P_SWEEP.
FULL = dict(
    test_shots={3: 500_000, 5: 1_000_000},
    val_shots=100_000,
    train_steps={3: 6_000, 5: 12_000},
    batch_size=4096,
    lr=1e-3,
    throughput_shots=100_000,
)
QUICK = dict(
    test_shots={3: 40_000, 5: 60_000},
    val_shots=10_000,
    train_steps={3: 600, 5: 1_000},
    batch_size=2048,
    lr=2e-3,
    throughput_shots=20_000,
)

SUB_BATCHES = 8          # each training batch = 8 sub-batches, each with its own random p
EVAL_POINTS = 30         # number of validation evaluations during training
TIMING_REPEATS = 5       # median of this many repeats for throughput timing


def get_mode(quick: bool) -> dict:
    """Return the knob dictionary for --quick or --full."""
    return QUICK if quick else FULL
