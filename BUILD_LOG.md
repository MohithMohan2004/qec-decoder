# Build Log

How this project was built, step by step, and the decisions made along the way. It is a hand-written record of the
process. The authoritative results are in `results/*.csv`, and the generated `report.md`, `DEMO_SCRIPT.md` and
`README.md` are built from them. Any number quoted here names the file it came from.

Specification followed: `task.md` (Sections 0-13).

---

## 1. Environment setup

- **Starting point:** system Python 3.12.3 with none of the required packages (no stim, pymatching, torch or
  scikit-learn). Laptop: Intel i5-10210U, 4 cores / 8 threads, 15 GB RAM, no GPU.
- **Decision:** nothing was installed system-wide. Everything went into a project virtualenv, `.venv`, with the
  **CPU-only PyTorch wheel** (`--index-url https://download.pytorch.org/whl/cpu`), which is much smaller than the CUDA
  build.
- All preferred packages installed, so none of the fallbacks in `task.md` (sklearn MLP instead of torch, CNN
  replacement) were needed.
- Installed versions are pinned in `requirements.txt` (generated from `pip freeze`) and recorded by every pipeline
  run in `results/env.txt`.

## 2. Base-paper check

- The paper *Rethink the Role of Neural Decoders in Quantum Error Correction* was found on arXiv (2605.12046;
  Ge Yan, Shanchuan Li, Yuxuan Du; listed as accepted to ICML 2026).
- Only the **abstract and introduction** were read (the PDF could not be read as text; the HTML version could).
  `report.md` summarises only those parts, in our own words, and says the rest was not reviewed (ground rule 5).

## 3. Code written (in build order)

| File | Purpose | Notes |
|---|---|---|
| `config.py` | Every knob in one place | `P_SWEEP`, seeds, QUICK/FULL settings, paths. `split_seed()` derives a distinct seed per (split, d, p). |
| `src/circuits.py` | Stim circuit builder | Rotated surface code, Z memory, circuit-level depolarizing noise p on gates, resets, measurements and idle data qubits. |
| `src/data.py` | Sampling + data plumbing | `sample()`, cached bit-packed val/test sets (`results/cache/*.npz`), `TrainStream` (fresh batches, 8 sub-batches each with a random p), rank-encoded grid map for the CNN. |
| `src/mwpm.py` | PyMatching baseline | Built from the detector error model with `decompose_errors=True`, separately for each (d, p). |
| `src/models.py` | MLP and CNN | Both take the flat detector vector. The CNN scatters it onto a [T, H, W] grid internally, so there is one input format. |
| `src/train.py` | Training loop | Adam, cosine LR decay, grad-clip 1.0, keeps the checkpoint with the best validation loss, logs a collapse warning, writes `.pt` + `.json`. |
| `src/evaluate.py` | Evaluation | LER + hand-written Wilson CI, throughput/latency, disagreement, generalisation, threshold estimate, sanity checks that stop the pipeline. |
| `src/plots.py` | 9 report figures | Okabe-Ito colour-blind-safe palette, decoders distinguished by colour and marker/line style, 200 dpi. |
| `src/viz.py` | Lattice picture | Plaquettes coloured by check type, detection events per round plus a collapsed view, ✓/✗ per decoder. |
| `tests/` | 14 unit tests | Detector counts (24 / 120), noiseless determinism, grid uniqueness and round-trip, seeding, PyMatching checks, Wilson edge cases, model shapes and sizes. |
| `run_all.py` | Pipeline | env check → tests → caches → train 5 models → evaluate → figures → docs; `--quick` / `--full` / `--skip-train`. |
| `demo.py` | Live demo | Runs from checkpoints only; `--d`, `--p`, `--seed`, `--no-plots`, `--pause`. |
| `make_report.py` | Docs generator | Writes `report.md`, `DEMO_SCRIPT.md`, `README.md` and `demo.ipynb` from the CSVs. |

### Design decisions worth knowing

- **One input format for both networks.** The CNN receives the same flat detector vector as the MLP and places it on
  the grid inside `forward()` using a registered index buffer. This keeps training, evaluation and the demo identical
  for both models.
- **Validation set size.** `val_shots` is treated as the *total* per distance, split evenly across the 8 noise
  levels. The fixed-p MLP is validated at p = 0.005 only, matching its training distribution.
- **Checkpoint selection** uses validation BCE loss (smoother than validation LER). The test set is never used for
  any training or selection decision.
- **Judgement sentences are computed.** Phrases such as "better", "tied" or "brittle" in the generated docs are
  chosen from the data (overlap of 95% CIs, ratio thresholds), not written in advance. This mattered: the quick and
  full runs reached different conclusions at d=3, and the documents updated automatically.
- **Single-shot latency for PyMatching** was added alongside the neural nets, for a like-for-like comparison.

## 4. Timing probe (before committing to the full run)

A quick benchmark of one training step at batch 4096 (in the session log, not saved to a file) gave about 0.15 s for
the CNN at d=3 and 0.42 s at d=5, with the MLPs and Stim sampling taking a few ms. That predicted roughly 2 hours of
training, inside the "few hours" limit in `task.md`, so **shots and steps were kept exactly as specified.** In
practice the CNNs ran about twice as slowly, because they share the CPU with validation passes (see section 6).

## 5. Quick run (`run_all.py --quick`)

- Took about 10 minutes. Tests passed, all sanity checks passed, and every network trained without collapsing to
  "always no flip". One early collapse warning for the d=5 CNN at step 33 resolved on its own.
- Used to debug the whole pipeline and review every figure. Fixes made after this run:
  - Fig 8 title was clipped, so the figure was made taller and the title split over three lines.
  - Report wording such as "drops by a factor of 0.4" was rewritten as "rises by a factor of 2.5".
  - Speed, brittleness and closing sentences in the demo script were made conditional on the data. The first draft
    assumed PyMatching was faster and that single-p training was brittle, and neither assumption held everywhere.
  - Empty categories ("worse at p = none") were removed from verdict sentences.

## 6. Full run (`run_all.py --full`)

- **Total 213.8 min** (`results/runtime.json`), of which training was 199.3 min. The d=5 CNN alone took 9,350 s,
  about 2.6 h (`checkpoints/cnn_d5.json`).
- All 5 models reached their best validation loss at or near the final step, so they had converged within the budget
  with no sign of overfitting. Fresh data at every step makes overfitting unlikely anyway.
- `SANITY OK: all checks passed` (`results/run_log.txt`).

## 7. Results summary

Full tables are in `report.md`; the values below are copied from `results/ler_results.csv` (p = 0.005).

| | PyMatching | MLP | CNN |
|---|---|---|---|
| d=3 LER | 0.0169 | 0.0161 (better) | 0.0155 (better) |
| d=5 LER | 0.0141 | 0.0294 (2.08× worse) | 0.0198 (1.40× worse) |

- **d=3:** both networks beat matching; the CNN is significantly better at all 8 noise levels. A likely reason is
  that matching treats X and Z errors as independent, while circuit-level noise produces correlated errors that a
  network can learn.
- **d=5:** both networks are significantly worse at every p, so the advantage reverses as the code grows.
- **Threshold** (PyMatching d=3/d=5 crossing): p ≈ 0.0063 (`results/threshold.csv`).
- **Speed at d=5** (`results/throughput.csv`): the MLP has the highest large-batch throughput, while PyMatching has
  by far the lowest single-shot latency. The CNN is the slowest in both.
- **Generalisation** (`results/generalisation.csv`): at d=3 an MLP trained only at p = 0.005 performs within about
  2% of the mixed-p MLP across the sweep.
- **Noise-model mismatch for matching:** weights for the wrong p change PyMatching's d=5 LER by under 2%.

## 8. Final verification (self-audit from `task.md` §12.3)

- [x] `pytest -q`: 14 passed.
- [x] Sanity checks passed in the full-run log.
- [x] `demo.py --d 3 --p 0.01 --no-plots` (2.3 s), with plots at d=3 (about 6 s) and at d=5 (6.4 s).
- [x] Every figure opened and checked for legibility, labels and overlapping text. Fig 4 was retitled after the full
      run, because "how much worse than matching" was no longer true at d=3.
- [x] No `TODO`, `XXX`, `placeholder` or `{` in `report.md` / `DEMO_SCRIPT.md`; only the `Team: ____` line remains.
- [x] Report numbers spot-checked against the CSVs (e.g. PyMatching d=3 at p=0.005 = 8,462 / 500,000 = 0.0169).
- [x] README has the overview, run commands, objective checklist, deviations, hardware and runtime.

## 9. Housekeeping after completion

- The project was first built in `quantum/qec-decoder/`. It was then flattened into the parent folder, which was
  renamed to `qec-decoder/`. `task.md` now sits next to the code.
- The virtualenv was rebuilt after the move (venvs store absolute paths) and `results/env.txt` was regenerated.
  Tests and the demo were re-run to confirm nothing broke.

## 10. What is left for the team

- Fill in the `Team: ____` line in `report.md`. It is generated, so edit `make_report.py` (search for `Team:`) or
  edit the file after the final regeneration.
- Rehearse with `DEMO_SCRIPT.md`; run `python demo.py --d 3 --p 0.01 --no-plots` once before presenting to warm up.
- Re-run `python make_report.py` after any new run so all documents stay in sync with `results/`.
