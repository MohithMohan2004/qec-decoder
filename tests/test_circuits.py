import numpy as np
import pytest

from src.circuits import make_circuit
from src.data import sample


@pytest.mark.parametrize("d,expected", [(3, 24), (5, 120)])
def test_detector_counts(d, expected):
    c = make_circuit(d, 0.005)
    assert c.num_detectors == expected
    assert c.num_observables == 1


@pytest.mark.parametrize("d", [3, 5])
def test_noiseless_is_deterministic(d):
    det, obs = sample(make_circuit(d, 0.0), 500, seed=1)
    assert not det.any()
    assert not obs.any()
