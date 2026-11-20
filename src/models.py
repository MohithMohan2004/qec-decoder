"""Small MLP and CNN decoders. Both take flat detection events [B, n_det] and output one logit."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

MLP_HIDDEN = {3: (128, 64), 5: (256, 128, 64)}


def make_mlp(n_in: int, hidden: tuple[int, ...]) -> nn.Sequential:
    """Fully connected net: n_in -> hidden... -> 1 logit, ReLU in between."""
    layers: list[nn.Module] = []
    width = n_in
    for h in hidden:
        layers += [nn.Linear(width, h), nn.ReLU()]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class MLPDecoder(nn.Module):
    def __init__(self, n_det: int, hidden: tuple[int, ...]):
        super().__init__()
        self.net = make_mlp(n_det, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CNNDecoder(nn.Module):
    """Scatter detectors onto a [T, H, W] grid (time as channels), then 3 conv layers + MLP head."""

    def __init__(self, flat_index: np.ndarray, grid_shape: tuple[int, int, int]):
        super().__init__()
        t, h, w = grid_shape
        self.grid_shape = grid_shape
        self.register_buffer("flat_index", torch.as_tensor(flat_index, dtype=torch.long))
        self.conv = nn.Sequential(
            nn.Conv2d(t, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * h * w, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, h, w = self.grid_shape
        g = x.new_zeros(x.shape[0], t * h * w)
        g[:, self.flat_index] = x
        return self.head(self.conv(g.view(-1, t, h, w))).squeeze(-1)


def build_model(kind: str, d: int, n_det: int, flat_index: np.ndarray, grid_shape: tuple[int, int, int]) -> nn.Module:
    """Factory used by training, evaluation and the demo."""
    if kind == "mlp":
        return MLPDecoder(n_det, MLP_HIDDEN[d])
    if kind == "cnn":
        return CNNDecoder(flat_index, grid_shape)
    raise ValueError(f"unknown model kind {kind!r}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
