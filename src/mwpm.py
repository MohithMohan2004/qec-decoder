"""PyMatching (minimum-weight perfect matching) baseline decoder."""
from __future__ import annotations

import numpy as np
import pymatching
import stim


def build_matcher(circuit: stim.Circuit) -> pymatching.Matching:
    """Matching graph whose edge weights come from the circuit's own detector error model."""
    dem = circuit.detector_error_model(decompose_errors=True)  # decomposition is required for matching
    return pymatching.Matching.from_detector_error_model(dem)


def decode(matcher: pymatching.Matching, det: np.ndarray) -> np.ndarray:
    """Predict the logical observable flip for each shot."""
    return matcher.decode_batch(det.astype(np.uint8))[:, 0].astype(bool)
