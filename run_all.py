"""Run the whole pipeline: env check -> tests -> caches -> training -> evaluation -> plots -> report.

    python run_all.py --quick    # ~10-15 min smoke test, demo-capable checkpoints
    python run_all.py --full     # the real numbers used by report.md
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import config


class Logger:
    """Print and append to results/run_log.txt."""

    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def env_check(log: Logger) -> None:
    py = sys.executable
    cmds = [
        [py, "--version"],
        [py, "-c", "import stim, pymatching, numpy, scipy, matplotlib, pandas; "
                   "print('stim', stim.__version__); print('pymatching', pymatching.__version__); "
                   "print('numpy', numpy.__version__); print('scipy', scipy.__version__); "
                   "print('matplotlib', matplotlib.__version__); print('pandas', pandas.__version__)"],
        [py, "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"],
        [py, "-c", "import sklearn; print('sklearn', sklearn.__version__)"],
        ["nproc"],
        ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo"],
    ]
    out = []
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True)
        out.append(f"$ {' '.join(c[:2]) if c[0] == py else ' '.join(c)}\n{r.stdout}{r.stderr}")
    (config.RESULTS / "env.txt").write_text("\n".join(out))
    log("environment recorded in results/env.txt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--quick", action="store_true")
    g.add_argument("--full", action="store_true")
    ap.add_argument("--skip-train", action="store_true", help="reuse existing checkpoints")
    a = ap.parse_args()
    quick = a.quick
    mode = config.get_mode(quick)
    for p in (config.RESULTS, config.CACHE, config.CHECKPOINTS, config.FIGURES):
        p.mkdir(parents=True, exist_ok=True)
    log = Logger(config.RESULTS / "run_log.txt")
    t0 = time.perf_counter()
    log(f"===== run_all {'--quick' if quick else '--full'} =====")

    log("STEP 1: environment check")
    env_check(log)

    log("STEP 2: tests")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.ROOT, capture_output=True, text=True)
    log(r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr)
    if r.returncode != 0:
        log(r.stdout + r.stderr)
        sys.exit("tests failed - stopping")

    log("STEP 3: building fixed val/test caches")
    from src.data import fixed_set, mixed_val_set
    for d in config.DISTANCES:
        mixed_val_set(d, mode["val_shots"])
        mixed_val_set(d, mode["val_shots"], [config.FIXED_P])
        for p in config.P_SWEEP:
            det, obs = fixed_set("test", d, p, mode["test_shots"][d])
            log(f"  test d={d} p={p:.3f}: {len(obs):,} shots, no-decoder flip rate {obs.mean():.5f}")

    from src.train import train
    if not a.skip_train:
        log("STEP 4: training")
        for kind, d, fp in [("mlp", 3, None), ("cnn", 3, None), ("mlp", 5, None), ("cnn", 5, None), ("mlp", 3, config.FIXED_P)]:
            meta = train(kind, d, quick, fp, log=log)
            log(f"  trained {meta['name']}: {meta['params']:,} params, best step {meta['best_step']}, {meta['train_seconds']}s")

    log("STEP 5: evaluation")
    from src import evaluate
    evaluate.run(quick, log)

    log("STEP 6: figures")
    from src import plots
    plots.make_all()

    log("STEP 7: report")
    import make_report
    make_report.main(make_figures=False)

    total = time.perf_counter() - t0
    (config.RESULTS / "runtime.json").write_text(json.dumps(
        {"mode": "quick" if quick else "full", "skip_train": a.skip_train, "total_seconds": round(total, 1)}, indent=2))
    log(f"===== done in {total / 60:.1f} min. See report.md, DEMO_SCRIPT.md, figures/ =====")
    print(make_report.headline())


if __name__ == "__main__":
    main()
