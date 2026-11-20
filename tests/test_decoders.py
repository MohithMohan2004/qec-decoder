import numpy as np
import torch

from src import mwpm
from src.circuits import make_circuit
from src.data import sample
from src.evaluate import wilson
from src.train import make_model


def test_pymatching_beats_no_decoder_at_low_p():
    c = make_circuit(3, 0.001)
    det, obs = sample(c, 50_000, seed=7)
    pred = mwpm.decode(mwpm.build_matcher(c), det)
    assert (pred != obs).mean() < obs.mean()


def test_pymatching_zero_errors_at_p0():
    c = make_circuit(3, 0.0)
    det, obs = sample(c, 1000, seed=7)
    # p=0 has no error mechanisms, so build the graph from a noisy circuit's structure
    pred = mwpm.decode(mwpm.build_matcher(make_circuit(3, 0.001)), det)
    assert (pred != obs).sum() == 0


def test_wilson_edge_cases():
    lo, hi = wilson(0, 1000)
    assert lo == 0.0 and 0 < hi < 0.01
    lo, hi = wilson(1000, 1000)
    assert hi == 1.0 and 0.99 < lo < 1.0
    lo, hi = wilson(50, 1000)
    assert lo < 0.05 < hi


def test_models_output_one_logit_per_shot():
    for d, n in [(3, 24), (5, 120)]:
        for kind in ("mlp", "cnn"):
            m = make_model(kind, d)
            assert m(torch.zeros(7, n)).shape == (7,)
            assert sum(p.numel() for p in m.parameters()) < 1_000_000
