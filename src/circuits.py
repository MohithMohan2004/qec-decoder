"""Stim circuit builders for the rotated surface-code memory experiment."""
from __future__ import annotations

import stim

import config


def make_circuit(d: int, p: float, rounds: int | None = None) -> stim.Circuit:
    """Rotated surface code, Z-basis memory, circuit-level depolarizing noise of strength p.

    Every Clifford gate, reset, measurement and idle data qubit (per round) gets noise p.
    """
    return stim.Circuit.generated(
        f"surface_code:{config.BASIS}",
        distance=d,
        rounds=rounds or config.rounds(d),
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def circuit_stats(circuit: stim.Circuit) -> dict:
    """Small summary used by the demo and the report."""
    return {
        "qubits": circuit.num_qubits,
        "detectors": circuit.num_detectors,
        "observables": circuit.num_observables,
        "measurements": circuit.num_measurements,
    }
