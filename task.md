# TASK: Surface-Code Decoding — PyMatching vs. a Small Neural Decoder

You are building a complete, demo-ready course project for **Quantum Computing (25AM3504)**.
Work autonomously from start to finish. Follow this file in order. Do not skip steps. Do not ask questions unless a step is impossible to complete (then state the blocker, pick the closest fallback listed here, and continue).

## 0. Project statement (the objectives you must fulfil)

Official project text:

> **Surface-code decoding: matching vs a small neural net.** Generate distance-3/5 surface-code data with Stim and compare PyMatching against a small MLP/CNN decoder.
> Base paper: *Rethink the Role of Neural Decoders in Quantum Error Correction*.

Objectives (each must be provably satisfied at the end, see Section 12 checklist):

- **O1** Generate surface-code memory-experiment data with **Stim** for **distance 3 and distance 5**.
- **O2** Build the **PyMatching** (minimum-weight perfect matching) baseline decoder.
- **O3** Build and train a **small MLP** decoder.
- **O4** Build and train a **small CNN** decoder.
- **O5** **Compare** all three decoders on logical error rate vs physical error rate, for both distances, with confidence intervals.
- **O6** Add a speed/size comparison (decode throughput, latency, parameter counts) and a short analysis of when neural decoders help or fail.
- **O7** Produce a **demo** that runs end to end in under 3 minutes from saved checkpoints, plus a word-for-word demo script.
- **O8** Produce a short **report** whose every number comes from saved result files.

Team context: a team of 3–4 students will present this. Everything must run on a laptop CPU. A GPU is optional.

## 1. Ground rules (read twice)

1. **Never invent numbers.** Every number in the report, README, demo script and plots must be read from files in `results/`. Write the report through a script (`make_report.py`) that reads the CSVs. No hand-typed results.
2. **Report honestly.** If the neural decoders lose to PyMatching, say so. That is a valid and expected finding. Do not tune on the test set to flatter the neural nets.
3. **Separate seeds.** Training data, validation data and test data use different, fixed seeds. Test data is never used for any training or model-selection decision.
4. **Reproducible.** One `config.py`, fixed seeds, `python run_all.py --quick` and `python run_all.py --full` must both work.
5. **Do not fabricate claims about the base paper.** If you can open the paper (network or local copy), read the abstract and intro and summarise only what you verified. If you cannot, cite it by title only and say in the report that its specific claims were not reviewed. Never attribute a result to it that you did not read.
6. **Environment.** Package installs may be blocked. First check what is installed (Section 2). Use the listed fallbacks instead of failing. If you could not use a preferred package, say so in `README.md` under "Deviations".
7. Keep code small, readable and commented. Prefer clarity to cleverness. Type hints on public functions. No dead code.

## 2. Step 1 — Environment check

Run and record the output in `results/env.txt`:

```bash
python --version
python -c "import stim, pymatching, numpy, scipy, matplotlib, pandas; print('stim', stim.__version__); print('pymatching', pymatching.__version__)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import sklearn; print('sklearn', sklearn.__version__)"
nproc
```

Required: `stim`, `pymatching` (>=2.0), `numpy`, `scipy`, `matplotlib`, `pandas`.
Preferred ML backend: `torch` (CPU is fine).
**Fallback if torch is missing and cannot be installed:** use `sklearn.neural_network.MLPClassifier` for the MLP, and for the CNN implement a small NumPy-free alternative by using `sklearn` MLP on spatially-structured features and clearly label it "CNN unavailable, convolutional model replaced by X" in the README. Only do this if torch is truly unavailable.
Optional: `pytest`, `tqdm`, `jupyter`/`nbformat`.

If installs are allowed: `pip install stim pymatching numpy scipy matplotlib pandas torch scikit-learn pytest tqdm --break-system-packages` (use the CPU torch wheel if possible).

## 3. Step 2 — Project layout

Create exactly this structure:

```
qec-decoder/
  README.md
  requirements.txt
  config.py
  run_all.py                 # runs the whole pipeline; flags --quick / --full
  demo.py                    # live demo from saved checkpoints (<3 min)
  make_report.py             # builds report.md + figures from results/*.csv
  DEMO_SCRIPT.md             # word-for-word talk track for the demo
  src/
    __init__.py
    circuits.py              # Stim circuit builders
    data.py                  # sampling, grid mapping, on-the-fly batch generator
    mwpm.py                  # PyMatching wrapper
    models.py                # MLP and CNN definitions
    train.py                 # training loop
    evaluate.py              # logical error rate, CIs, timing
    plots.py                 # all figures
    viz.py                   # lattice/syndrome visualisation for the demo
  tests/
    test_circuits.py
    test_data.py
    test_decoders.py
  results/                   # CSVs, logs, env.txt
  checkpoints/               # trained model weights (.pt) + metadata json
  figures/                   # PNGs used by report and demo
```

Write `requirements.txt` with the versions actually installed (from `pip freeze` filtered to the packages used).

## 4. Step 3 — `config.py`

One place for every knob. Use a dataclass or plain constants.

```python
DISTANCES = [3, 5]
ROUNDS = lambda d: d                      # memory experiment lasts d rounds
BASIS = "rotated_memory_z"
P_SWEEP = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012]  # physical error rates

SEEDS = {"train": 1000, "val": 2000, "test": 3000}

FULL = dict(
    test_shots={3: 500_000, 5: 1_000_000},
    val_shots=100_000,
    train_steps={3: 6_000, 5: 12_000},
    batch_size=4096,
    lr=1e-3,
)
QUICK = dict(
    test_shots={3: 40_000, 5: 60_000},
    val_shots=10_000,
    train_steps={3: 600, 5: 1_000},
    batch_size=2048,
    lr=2e-3,
)
```

Adjust step counts only if training clearly has not converged (validation loss still falling) or takes unreasonably long. Record any change and reason in the README.

## 5. Step 4 — Data generation with Stim (O1)

File `src/circuits.py`:

```python
import stim

def make_circuit(d: int, p: float, rounds: int | None = None) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds or d,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )
```

This is a circuit-level depolarizing noise model: every gate, reset, measurement and idle data qubit gets noise of strength `p`.

File `src/data.py` must provide:

```python
def sample(circuit, shots, seed):
    """Return (detection_events [shots, n_det] uint8/bool, observable_flips [shots] bool)."""
    sampler = circuit.compile_detector_sampler(seed=seed)   # if seed kwarg unsupported, fall back to no seed and np.random.seed
    det, obs = sampler.sample(shots, separate_observables=True)
    return det, obs[:, 0]
```

Rules:
- **Never hard-code the number of detectors.** Read `circuit.num_detectors`. (Expected as a sanity check: d=3 has 24 detectors, d=5 has 120 for `rounds=d`. Assert this in tests.)
- **Training data is generated on the fly**: each training step samples a fresh batch from Stim, with `p` drawn uniformly from `P_SWEEP` per sub-batch. Precompile one sampler per `(d, p)` once. This avoids overfitting and avoids storing millions of rows.
- **Validation and test sets are fixed**: sampled once per `(d, p)` with their own seeds and cached to `results/cache/*.npz` (gitignore-able).
- Report the **baseline "always predict no flip" error rate** for each `(d, p)`; it is the floor any decoder must beat.

### 5.1 Grid mapping for the CNN

Stim gives each detector coordinates `(x, y, t)` via `circuit.get_detector_coordinates()`. Build a deterministic map to a tensor of shape `[T, H, W]` (time steps as channels, spatial 2D grid) by **rank-encoding** each coordinate (rank among the sorted unique values), so it does not depend on coordinate scaling:

```python
def build_grid_index(circuit):
    coords = circuit.get_detector_coordinates()            # {det_index: [x, y, t]}
    idx = sorted(coords)                                   # 0..n_det-1
    xs = sorted({coords[i][0] for i in idx}); ys = sorted({coords[i][1] for i in idx}); ts = sorted({coords[i][2] for i in idx})
    xr = {v: k for k, v in enumerate(xs)}; yr = {v: k for k, v in enumerate(ys)}; tr = {v: k for k, v in enumerate(ts)}
    ti = [tr[coords[i][2]] for i in idx]; yi = [yr[coords[i][1]] for i in idx]; xi = [xr[coords[i][0]] for i in idx]
    return np.array(ti), np.array(yi), np.array(xi), (len(ts), len(ys), len(xs))

def to_grid(det, index):
    ti, yi, xi, shape = index
    g = np.zeros((det.shape[0], *shape), dtype=np.float32)
    g[:, ti, yi, xi] = det
    return g
```

Assert that `(t, y, x)` triples are unique (no two detectors collide). The grid is sparse; that is fine.

## 6. Step 5 — PyMatching baseline (O2)

File `src/mwpm.py`:

```python
import pymatching

def build_matcher(circuit):
    dem = circuit.detector_error_model(decompose_errors=True)   # decompose_errors=True is REQUIRED for matching
    return pymatching.Matching.from_detector_error_model(dem)

def decode(matcher, det):
    return matcher.decode_batch(det.astype("uint8"))[:, 0].astype(bool)
```

Build the matcher **per `(d, p)`** so that edge weights match the noise level (this is the standard, strong baseline). Add an extra variant, **"PyMatching (fixed p=0.005 weights)"**, to show sensitivity to a mis-specified noise model. This variant must be a clearly labelled second line in the plots.

Test (`tests/test_decoders.py`): at `d=3, p=0.001`, PyMatching's logical error rate must be below the always-no-flip rate; at `p=0` it must be exactly 0.

## 7. Step 6 — Neural decoders (O3, O4)

Both decoders are **binary classifiers**: input = detection events from one shot, output = one logit for "the logical observable flipped". Loss: `BCEWithLogitsLoss`. Prediction: `logit > 0`.

File `src/models.py`:

**MLP** (small):
- d=3 (24 inputs): `Linear(24,128) → ReLU → Linear(128,64) → ReLU → Linear(64,1)`
- d=5 (120 inputs): `Linear(120,256) → ReLU → Linear(256,128) → ReLU → Linear(128,64) → ReLU → Linear(64,1)`
- Build with a helper `make_mlp(n_in, hidden=(…))`.

**CNN** (small): input `[B, T, H, W]` from the grid map, treating time as channels:
```
Conv2d(T, 32, 3, padding=1) → ReLU
Conv2d(32, 64, 3, padding=1) → ReLU
Conv2d(64, 64, 3, padding=1) → ReLU
Flatten → Linear(64*H*W, 128) → ReLU → Linear(128, 1)
```
Keep each model under ~1M parameters. Log exact parameter counts to `results/model_sizes.csv`.

If a CNN underperforms, you may add: batch-norm, a second pooling-free block, or a residual connection. Keep it "small" and document any change.

File `src/train.py`:
- Optimizer Adam, cosine learning-rate decay, gradient clipping at 1.0.
- Each step: fresh Stim batch (mixed `p` from `P_SWEEP`), convert to tensor, update.
- Every N steps: evaluate on the fixed **validation** set (mixed p) and log `step, train_loss, val_loss, val_ler` to `results/train_log_{model}_d{d}.csv`.
- Keep the **best validation** checkpoint. Save `checkpoints/{model}_d{d}.pt` plus `checkpoints/{model}_d{d}.json` (params count, steps, seeds, git-less timestamp, config used).
- Set `torch.manual_seed`, `np.random.seed`. Use `torch.set_num_threads(os.cpu_count())`.
- **Known pitfall:** at low `p` almost all syndromes are empty and a network can collapse to "always predict 0". Mixed-`p` training (including the higher values) plus a larger batch size avoids this. Log the fraction of predicted flips on validation as a health check and raise a warning if it is exactly 0 while the true rate is above 1%.

CLI: `python -m src.train --model mlp --d 3 [--quick]`.

## 8. Step 7 — Evaluation (O5, O6)

File `src/evaluate.py`. For each `d ∈ {3,5}`, each `p ∈ P_SWEEP`, on the fixed **test** set:

Decoders: `pymatching`, `pymatching_fixed_p`, `mlp`, `cnn`, `no_decoder` (always predict no flip).

Metrics, written to `results/ler_results.csv` with columns
`distance, p, decoder, shots, errors, ler, ci_low, ci_high`:

- Logical error rate per shot: `errors / shots`.
- **95% Wilson score interval** (implement by hand, no extra dependency). It must behave sensibly when `errors == 0`.
- Also compute per-round LER: `1 - (1 - ler)^(1/rounds)` as an extra column `ler_per_round`.

Further analyses (each its own CSV):

1. `results/throughput.csv` — decode time per shot and shots/second for each decoder, `d ∈ {3,5}`, at `p=0.005`:
   - PyMatching, single process, batch of 100k shots.
   - Neural nets on CPU: batch size 1 (latency) and batch size 4096 (throughput).
   - Use `time.perf_counter`, warm-up run first, median of 5 repeats. State the hardware (CPU model, threads) in the file header or `env.txt`.
2. `results/model_sizes.csv` — parameter counts.
3. `results/disagreement.csv` — at `p=0.005`: fraction of shots where each neural decoder disagrees with PyMatching, and who is right when they disagree (counts of: NN right / MWPM right / both wrong).
4. `results/generalisation.csv` — **train-on-one-p, test-on-others**: train one extra MLP for `d=3` at fixed `p=0.005` only (not mixed), then evaluate across `P_SWEEP` and compare with the mixed-`p` MLP. Shows how brittle a neural decoder is to noise-model mismatch, versus PyMatching which reweights from the error model.
5. `results/threshold.csv` — estimate the PyMatching crossing point where the d=3 and d=5 LER curves intersect (linear interpolation of the log-LER difference). Report the estimate and the bracketing `p` values. If no crossing is visible in the sweep, say so and extend `P_SWEEP` upward; do not fake one.

### Sanity checks that MUST pass before you trust any result

- PyMatching `d=5` LER < `d=3` LER at `p ≤ 0.004` (below threshold, larger code is better). Above ≈1% the order should flip or converge. If this fails, the circuit or decoder is wrong: debug before continuing.
- LER is monotonically non-decreasing in `p` for PyMatching (allow tiny statistical noise).
- Every neural decoder beats `no_decoder` at the highest `p`. If it does not, training failed: fix and retrain.
- `errors <= shots` and all CIs contain the point estimate.

Write these as assertions in `evaluate.py` that log a loud `SANITY FAIL` message and stop the pipeline.

## 9. Step 8 — Figures (`src/plots.py`, saved to `figures/`)

All plots: matplotlib, readable fonts (≥12 pt), labelled axes with units, legend, grid, saved at 200 dpi PNG. Colour-blind-safe palette; distinguish decoders by both colour **and** marker/line style.

1. `fig1_ler_vs_p_d3.png` and `fig2_ler_vs_p_d5.png` — log-y LER vs `p` for all decoders with 95% CI error bars/bands.
2. `fig3_threshold.png` — PyMatching d=3 vs d=5 on one axis with the estimated crossing marked.
3. `fig4_ratio.png` — LER ratio `decoder / pymatching` vs `p` for MLP and CNN, both distances, with a dashed line at 1.0.
4. `fig5_training_curves.png` — validation loss vs step for all four trained models.
5. `fig6_throughput.png` — bar chart of shots/s and latency (log scale).
6. `fig7_generalisation.png` — mixed-`p` MLP vs fixed-`p` MLP vs PyMatching across `P_SWEEP` (d=3).
7. `fig8_disagreement.png` — stacked bars of who is right when decoders disagree.
8. `fig9_scaling.png` — gap of each neural decoder to PyMatching at d=3 vs d=5 (the "does it scale" picture).

Every figure title must state the noise model ("circuit-level depolarizing, rounds = d") and shot counts.

## 10. Step 9 — Demo (O7)

### 10.1 `src/viz.py` — lattice view
Draw the rotated surface-code patch for `d=3` (or `d=5` via flag), with detector events from one sampled shot highlighted on the lattice (collapsing time, or showing a slider/grid of rounds as small multiples). Annotate: true logical flip (yes/no) and each decoder's prediction with a ✓/✗. Save as `figures/demo_shot_*.png`.

### 10.2 `demo.py`
Runs **from saved checkpoints only** (no training), in under 3 minutes on a laptop. Steps, each printed with a clear header:

1. Print the Stim circuit stats (qubits, detectors, d, rounds) for the chosen distance.
2. Sample one shot at `p=0.01`; show its syndrome and draw it with `viz.py`.
3. Decode with PyMatching, MLP, CNN; show predictions vs truth.
4. Run a mini-benchmark (20k shots per decoder) at three values of `p` and print a table.
5. Display the main comparison figure (`fig1`/`fig2`) and the throughput bar chart.
6. Print a 3-line conclusion **computed from `results/*.csv`**, not hard-coded.

Flags: `--d {3,5}`, `--p FLOAT`, `--seed INT`, `--no-plots` (print only), `--pause` (wait for Enter between sections, useful when presenting). If checkpoints are missing, print a clear message telling the user to run `python run_all.py --quick`.

### 10.3 `DEMO_SCRIPT.md` — word-for-word talk track
Write a script for a **5-minute presentation**, with the exact sentences to say and the exact command or slide to show at each moment, formatted as a two-column table `[Time | Say | Show/Run]`. Required beats, in order:

1. (0:00) One-sentence hook: quantum computers make errors constantly; error correction is the thing that makes them useful; decoding is the fast classical step that decides how to fix them.
2. (0:30) What a surface code and a syndrome are, in plain language, with the lattice picture.
3. (1:15) The problem: given the syndrome, decide if the logical qubit flipped. Two approaches: PyMatching (a graph algorithm that pairs up defects with minimum total weight) vs a neural network that learns the answer from examples.
4. (2:00) How we built it: Stim for data, circuit-level noise, d=3 and d=5, small MLP and CNN, trained on fresh samples.
5. (2:45) **Live demo** — run `python demo.py --d 3 --p 0.01 --pause`, narrating each section.
6. (4:00) Results: read **the actual numbers from `results/` and fill them into the script via `make_report.py`**; the script must contain real values (e.g. "At p = X, PyMatching gets Y and the CNN gets Z") generated from the CSVs, not placeholders.
7. (4:30) Honest takeaway: where neural decoders match matching, where they lose, speed/size trade-off, brittleness to noise-model mismatch, and what we would do next.
8. (4:50) Closing line and a call for questions.

Also add a **"Likely questions & answers"** section with at least 8 Q&As (for example: why circuit-level noise; what is a threshold; why does PyMatching win or lose at d=5; could the network see more rounds than it was trained on; why mixed-p training; what is a Wilson interval; why is decoder speed important; what would a real-time decoder need). Every numeric answer must come from results files.

`DEMO_SCRIPT.md` is generated by `make_report.py` from a template so its numbers are filled from CSVs.

### 10.4 Optional notebook
If `nbformat` is available, also generate `demo.ipynb` that mirrors `demo.py` with markdown explanations. Skip silently if not.

## 11. Step 10 — Report (`make_report.py` → `report.md`, O8)

Structure (target 3–5 pages equivalent):

1. **Title, team placeholder** (`Team: ____`), course code.
2. **Background**: surface code, syndrome, decoding, MWPM, neural decoders. Cite the base paper by title; summarise it only if verified (Ground rule 5).
3. **Method**: circuit, noise model, data generation, models (with parameter counts from CSV), training protocol, evaluation protocol, statistics (Wilson CI).
4. **Results**: embed figures; tables auto-generated from CSVs (LER at `p ∈ {0.003, 0.005, 0.008}` for each decoder and distance, estimated threshold, throughput, parameters).
5. **Analysis**: answer explicitly — Do neural decoders match PyMatching at d=3? At d=5? Where do they fall behind? How much do they lose when the training noise differs from the test noise? How do speed and size compare? Base every sentence on numbers present in the CSVs; hedge where CIs overlap (say "not statistically distinguishable").
6. **Limitations**: simple memory experiment only, one noise model, no measurement-error-aware sliding windows, small models, CPU-only timings, finite statistics.
7. **Future work**: larger distances, graph neural networks or transformers, correlated noise, hybrid "NN + matching" decoders, real-time FPGA considerations.
8. **Reproducibility**: commands, seeds, hardware, package versions.
9. **References**: list the base paper title, Stim, PyMatching (Higgott & Gidney), and the surface-code memory experiment source only as names/titles you are certain of. No invented URLs, DOIs or years.

## 12. Step 11 — Tests, pipeline, and the Definition of Done

### 12.1 Tests (`pytest -q` must pass)
- `test_circuits.py`: detector counts for d=3/5, one observable, circuit is deterministic at `p=0` (all detectors zero).
- `test_data.py`: grid mapping shape and uniqueness; `to_grid` round-trips detector values; same seed gives identical samples.
- `test_decoders.py`: items from Section 6 and Wilson-interval edge cases (0 errors, all errors).

### 12.2 `run_all.py`
Order: env check → tests → build caches → train (mlp d3, cnn d3, mlp d5, cnn d5, mlp d3 fixed-p) → evaluate → plots → `make_report.py` → print final summary. Log to `results/run_log.txt`. Support `--quick` (≈10–15 min, for smoke-testing and demo checkpoints) and `--full` (the real numbers). Run `--quick` first, fix any problems, then run `--full`. The demo and report must be built from the **full** results.

If `--full` would take longer than a few hours on this machine, reduce shots/steps proportionally, state the reduction in the README, and keep the experiment design identical.

### 12.3 Definition of Done — verify and tick each item in `README.md` under "Objective checklist"

| ID | Requirement | Evidence file |
|----|-------------|---------------|
| O1 | Stim data for d=3 and d=5 | `src/circuits.py`, `results/cache/`, tests pass |
| O2 | PyMatching baseline | `src/mwpm.py`, rows in `ler_results.csv` |
| O3 | Small MLP trained | `checkpoints/mlp_d{3,5}.pt`, `train_log_*` |
| O4 | Small CNN trained | `checkpoints/cnn_d{3,5}.pt`, `train_log_*` |
| O5 | LER comparison with CIs | `ler_results.csv`, `fig1`, `fig2`, `fig4` |
| O6 | Speed/size + analysis | `throughput.csv`, `model_sizes.csv`, `disagreement.csv`, `generalisation.csv`, `threshold.csv` |
| O7 | Demo + word-for-word script | `demo.py` runs <3 min, `DEMO_SCRIPT.md` |
| O8 | Report from data | `report.md` generated, no hand-typed numbers |

Final self-audit (do this last, and fix anything that fails):
- [ ] `pytest -q` passes.
- [ ] All Section 8 sanity checks passed in the log.
- [ ] `python demo.py --d 3 --p 0.01 --no-plots` finishes without error, then the same with plots and `--d 5`.
- [ ] Open every PNG in `figures/` and confirm it is legible, correctly labelled, and has no overlapping text.
- [ ] Grep `report.md` and `DEMO_SCRIPT.md` for the strings `TODO`, `XXX`, `{`, `placeholder`: none may remain (except the `Team: ____` line).
- [ ] Spot-check at least five numbers in the report against the CSVs.
- [ ] `README.md` has: what the project is, how to run quick/full/demo, the objective checklist, deviations, hardware used, and total runtime.

## 13. Final response format

When finished, reply with a short summary only:
- What was built and where (the folder path).
- The headline result in 2–3 sentences, with numbers taken from the CSVs.
- Any deviations or limitations (missing packages, reduced shots).
- The exact command to run the demo.

Do not paste the code or the whole report into the reply.
