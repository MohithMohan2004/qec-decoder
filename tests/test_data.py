import numpy as np
import pytest

from src.circuits import make_circuit
from src.data import TrainStream, build_grid_index, flat_grid_index, sample, to_grid


@pytest.mark.parametrize("d,shape", [(3, (4, 4, 4)), (5, (6, 6, 6))])
def test_grid_shape_and_uniqueness(d, shape):
    ti, yi, xi, s = build_grid_index(make_circuit(d, 0.005))
    assert s == shape
    assert len(set(zip(ti, yi, xi))) == len(ti)
    assert len(set(flat_grid_index((ti, yi, xi, s)))) == len(ti)


@pytest.mark.parametrize("d", [3, 5])
def test_to_grid_round_trip(d):
    c = make_circuit(d, 0.01)
    index = build_grid_index(c)
    det, _ = sample(c, 200, seed=5)
    g = to_grid(det, index)
    ti, yi, xi, _ = index
    np.testing.assert_array_equal(g[:, ti, yi, xi].astype(bool), det)
    assert g.sum() == det.sum()  # nothing written outside detector cells


def test_same_seed_same_samples():
    c = make_circuit(3, 0.01)
    a = sample(c, 1000, seed=42)
    b = sample(c, 1000, seed=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_train_stream_batch_shape():
    det, obs = TrainStream(3, [0.005, 0.01], seed=0).next_batch(256)
    assert det.shape == (256, 24) and obs.shape == (256,)
