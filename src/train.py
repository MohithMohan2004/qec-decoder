"""Training loop for the neural decoders.

CLI:  python -m src.train --model mlp --d 3 [--quick] [--fixed-p 0.005]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import warnings
from datetime import datetime

import numpy as np
import torch
from torch import nn

import config
from src.circuits import make_circuit
from src.data import TrainStream, build_grid_index, flat_grid_index, mixed_val_set
from src.models import build_model, count_params


def model_name(kind: str, d: int, fixed_p: float | None = None) -> str:
    """File stem for checkpoints and logs, e.g. 'mlp_d3' or 'mlp_fixedp_d3'."""
    return f"{kind}_fixedp_d{d}" if fixed_p is not None else f"{kind}_d{d}"


def make_model(kind: str, d: int) -> nn.Module:
    """Build an untrained model with the right input size / grid for distance d."""
    circuit = make_circuit(d, 0.0)
    index = build_grid_index(circuit)
    return build_model(kind, d, circuit.num_detectors, flat_grid_index(index), index[3])


@torch.no_grad()
def predict_logits(model: nn.Module, det: np.ndarray, batch: int = 65_536) -> np.ndarray:
    """Logits for a large detector array, evaluated in chunks."""
    model.eval()
    out = [model(torch.from_numpy(det[i:i + batch].astype(np.float32))).numpy() for i in range(0, len(det), batch)]
    return np.concatenate(out)


def train(kind: str, d: int, quick: bool, fixed_p: float | None = None, log=print) -> dict:
    """Train one decoder, keep the best-validation-loss checkpoint, return its metadata."""
    mode = config.get_mode(quick)
    torch.manual_seed(config.SEEDS["train"] + d)
    np.random.seed(config.SEEDS["train"] + d)
    torch.set_num_threads(os.cpu_count() or 1)

    ps = [fixed_p] if fixed_p is not None else config.P_SWEEP
    name = model_name(kind, d, fixed_p)
    steps, bs = mode["train_steps"][d], mode["batch_size"]
    stream = TrainStream(d, ps, seed=config.SEEDS["train"] + 10 * d + (500 if fixed_p else 0))
    val_det, val_obs = mixed_val_set(d, mode["val_shots"], ps)
    val_x = val_det.astype(np.float32)
    true_rate = float(val_obs.mean())

    model = make_model(kind, d)
    opt = torch.optim.Adam(model.parameters(), lr=mode["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    loss_fn = nn.BCEWithLogitsLoss()
    eval_every = max(steps // config.EVAL_POINTS, 10)

    config.RESULTS.mkdir(exist_ok=True)
    config.CHECKPOINTS.mkdir(exist_ok=True)
    log_path = config.RESULTS / f"train_log_{name}.csv"
    ckpt_path = config.CHECKPOINTS / f"{name}.pt"
    best = math.inf
    best_step = 0
    running = []
    t0 = time.perf_counter()
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "train_loss", "val_loss", "val_ler", "val_pred_flip_frac", "val_true_flip_frac", "lr", "elapsed_s"])
        for step in range(1, steps + 1):
            model.train()
            det, obs = stream.next_batch(bs)
            x = torch.from_numpy(det.astype(np.float32))
            y = torch.from_numpy(obs.astype(np.float32))
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running.append(loss.item())

            if step % eval_every == 0 or step == steps:
                logits = predict_logits(model, val_x)
                val_loss = float(loss_fn(torch.from_numpy(logits), torch.from_numpy(val_obs.astype(np.float32))))
                pred = logits > 0
                val_ler = float((pred != val_obs).mean())
                pred_frac = float(pred.mean())
                if pred_frac == 0.0 and true_rate > 0.01:
                    warnings.warn(f"{name}: network predicts no flips at all (true flip rate {true_rate:.3f}) - collapsed?")
                elapsed = time.perf_counter() - t0
                w.writerow([step, f"{np.mean(running):.6f}", f"{val_loss:.6f}", f"{val_ler:.6f}",
                            f"{pred_frac:.5f}", f"{true_rate:.5f}", f"{sched.get_last_lr()[0]:.2e}", f"{elapsed:.1f}"])
                f.flush()
                running = []
                if val_loss < best:
                    best, best_step = val_loss, step
                    torch.save(model.state_dict(), ckpt_path)
                log(f"  [{name}] step {step:>6}/{steps}  val_loss {val_loss:.4f}  val_ler {val_ler:.4f}  "
                    f"pred_flip {pred_frac:.3f}/{true_rate:.3f}  {elapsed:.0f}s")

    meta = {
        "name": name, "kind": kind, "distance": d, "params": count_params(model),
        "steps": steps, "best_step": best_step, "best_val_loss": best,
        "train_ps": ps, "batch_size": bs, "lr": mode["lr"], "mode": "quick" if quick else "full",
        "seeds": config.SEEDS, "train_seconds": round(time.perf_counter() - t0, 1),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (config.CHECKPOINTS / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return meta


def load_trained(kind: str, d: int, fixed_p: float | None = None) -> nn.Module:
    """Load a saved checkpoint (raises FileNotFoundError if missing)."""
    path = config.CHECKPOINTS / f"{model_name(kind, d, fixed_p)}.pt"
    model = make_model(kind, d)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=["mlp", "cnn"], required=True)
    ap.add_argument("--d", type=int, choices=config.DISTANCES, required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fixed-p", type=float, default=None, help="train at a single p instead of the mixed sweep")
    a = ap.parse_args()
    print(json.dumps(train(a.model, a.d, a.quick, a.fixed_p), indent=2))
