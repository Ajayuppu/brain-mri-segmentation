# Replication guide

This guide is the public-repo path for repeating the LGG FLAIR binary-segmentation study: environment setup, the Kaggle dataset, the data-root layout the scripts actually read, the metadata audit, the validation always-empty reference, the 14 training runs, the one-shot freeze, and the single test evaluation. It was written from the code and docs in this repository.

Checkpoints, evaluation evidence, and the written report are not in this repository. They are kept in a private companion repository and are available on request. This file does not link that repository, and it does not republish its numeric test table.

The study that those artifacts record was completed, frozen, and test-evaluated once on 2026-11-08.

## Hardware

The study machine was an NVIDIA L40S. `docs/PROTOCOL_AUDIT_20261108.md` records the environment taken from that machine's run metadata: Python 3.12.15, PyTorch 2.8.0, a PyTorch build against CUDA 12.9, and NVIDIA driver 595.112.07 (the driver reported CUDA 13.2). A comparison that is meant to match the study should be trained on a CUDA GPU. CPU PyTorch is enough for `pytest` and for the smoke command in the README, and it is not a substitute for the 50-epoch runs.

`scripts/train.py` records `torch.cuda.get_device_name`, `nvidia-smi` (name, driver version, total memory), torch/CUDA/cuDNN versions, and whether cuDNN benchmark and deterministic mode were on. It does not turn deterministic mode on. See [Nondeterminism](#nondeterminism).

## Setup

Python 3.10 or newer. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs a CPU-capable PyTorch plus numpy, pandas, tifffile, pyyaml, pillow, matplotlib, and pytest. On a machine whose image already provides PyTorch and CUDA, install only the extra packages and do not replace that torch build:

```bash
pip install -r requirements-gpu.txt
```

Run every command below from the repository root so `configs/` and the default `runs/` and `FROZEN.json` paths resolve. The package root used for freeze hashes is the repository root (`src/lggseg/protocol.py` walks two directories up).

Synthetic tests do not need the dataset or a GPU:

```bash
pytest -q tests
```

## Dataset

Download the public Kaggle dataset [mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Mateusz Buda; FLAIR slices and segmentation masks from TCIA TCGA-LGG). Citation and the CC BY-NC-SA 4.0 terms are in the README. This repository does not redistribute the images or the masks.

Keep the download outside git. `data/`, `runs/`, `evidence/`, checkpoints, and `FROZEN.json` are gitignored.

### Known gap: nothing builds the manifests from the Kaggle zip

No script in this repository converts the Kaggle archive into `manifest.csv`, `patient_splits.csv`, the split manifests, or the `train/`, `val/`, and `test/` trees. The patient-wise partition is an input, not something this code constructs. This guide does not define a split procedure. Do not invent one (random slice split, a new patient split, a split that only matches the patient counts below, or a split read out of the test labels).

Training and the audit need a data root that already has the schema below. That prepared root is distributed with the course materials and can be requested with the private companion. If you do not have it, stop before `audit_metadata.py`. Pointing `--data-root` at the raw Kaggle zip, or at the upstream folder layout, will fail: the loaders read the CSV names below and nothing else.

### Data-root schema

`--data-root` is a directory (the README uses `data/`) containing:

```text
data/
  manifest.csv
  patient_splits.csv
  train_manifest.csv
  val_manifest.csv
  test_manifest.csv
  train/
  val/
  test/
```

`image_path` and `mask_path` are relative to that directory. Images are 256×256×3 TIFFs, read with `tifffile` in stored channel order (HWC, last dimension 3) and cast to float32. Masks are stored as 0/255. `read_mask` takes the first channel when the TIFF is 3D, then binarizes with `> 0`.

`scripts/audit_metadata.py` reads these columns:

| File | Columns the audit or loaders use |
| --- | --- |
| `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv` | `split`, `patient_id`, `slice_id`, `image_path`, `mask_path`, `height`, `width`, `channels`, `image_sha256`, `has_foreground`, `foreground_pixels` |
| `manifest.csv` | same row schema; the audit uses it as the full-manifest row count |
| `patient_splits.csv` | `patient_id`, `pair_count` |

`load_manifest` requires every row's `split` value to equal the file it was loaded from (`train`, `val`, or `test`). Training uses `patient_id`, `slice_id`, `image_path`, `mask_path`, and, for `weighted_bce`, `foreground_pixels`. Evaluation uses `patient_id`, `slice_id`, `image_path`, and `mask_path`. The audit also requires `height`, `width`, `channels`, `image_sha256`, and `has_foreground`.

`has_foreground` and `foreground_pixels` on `test_manifest.csv` are label-derived. Treat that file as holdout metadata: do not use it to choose a method, a threshold, or a split. `scripts/train.py` opens only `train_manifest.csv` and `val_manifest.csv`.

Checks `audit_metadata.py` computes (it writes them; it does not exit non-zero when they fail):

- Per split: patient count, row count, `split` column matches the file, every row is `height == 256`, `width == 256`, `channels == 3`, and the number of duplicated `image_sha256` values.
- Patient-set intersection sizes for train–val, train–test, and val–test. A prepared root that matches the study has empty intersections.
- Row count of the three split files equals the row count of `manifest.csv`. This is a count check, not a row-by-row identity check.
- No duplicated `image_sha256` across the concatenated split files.
- `patient_splits.csv` `pair_count` equals the number of rows for that `patient_id` in the concatenated split files.
- Train and val only: positive and empty slice counts, foreground fraction, foreground-pixel quantiles on positive slices, count of positive slices under 100 pixels, and per-patient positive fraction. Test label statistics are not computed.

The same script records these expected sizes and does not compare them itself. After a run, compare `checks.<split>.patients` and `checks.<split>.pairs` to:

| Split | Patients | Slices (pairs) |
| --- | --- | --- |
| train | 77 | 2763 |
| val | 17 | 598 |
| test | 16 | 568 |

Those three numbers per split are the study sizes stored in `scripts/audit_metadata.py`. They are not instructions for how to assign patients. This repository does not contain the code that filled `image_sha256`; the audit only detects duplicate values in that column.

`test_manifest.csv` must be present at audit time because the script reads it. Reading it is a metadata check. Do not open the test images or masks until the freeze and the single evaluation below.

## Shared training budget

Every file in `configs/` except `configs/pilot_bce_3ep.yaml` uses the same budget. Only the loss changes. The pilot config is a 3-epoch throughput check (`name: PILOT_bce_3ep`). It is marked smoke-test if you pass `--max-train`, and `freeze` rejects any run whose `run_meta.json` has `smoke_test: true`. Do not include it in the 14-run study.

| Setting | Value |
| --- | --- |
| Epochs | 50 |
| Batch size | 16 |
| Optimizer | AdamW, `lr` 0.001, `weight_decay` 0.0001 |
| Schedule | cosine annealing over 50 epochs |
| Model | 2D U-Net from scratch, `base_channels` 32, depth 4, 3 input channels, 1 logit channel, no pretrained weights |
| Workers | 4 |
| `drop_last` | true on the training loader |
| Selection metric | `pooled_foreground_dice` on the validation set |
| Threshold grid | 0.05, 0.10, …, 0.95 (step 0.05) |
| Augmentation | train only; horizontal flip with probability 0.5, then `rot90` by k in {0, 1, 2, 3}; the same draw is applied to the image and the mask |
| Augmentation RNG | NumPy generator seeded by `(seed, epoch, index)` |

Checkpoint selection, from `run_meta.json`'s `selection_rule`: the checkpoint is the epoch that maximizes the max-over-grid validation metric, and the threshold is re-selected on that checkpoint. The grid search is validation-only. The headline test metrics (positive-slice Dice and empty-slice false-positive rate) are not the selection metric.

Normalization is per-channel mean and standard deviation on training-image pixels only (population second moment, variance floored at `1e-12` before the square root). The first full training run writes `runs/train_channel_stats.json`. Later full runs reuse that file when it exists, and each run copies the stats to `runs/<name>_s<seed>/channel_stats.json` with `fitted_on` equal to `train`. Start from an empty `runs/` directory. A `--max-train` smoke run does not write the shared stats file. `freeze` rejects stats that are not `fitted_on: train`.

Losses (`src/lggseg/losses.py`):

| Config | `loss` | What is optimized |
| --- | --- | --- |
| `configs/bce_baseline.yaml` | `bce` | `BCEWithLogits` |
| `configs/bce_dice.yaml` | `bce_dice` | BCE + 1.0 × soft Dice, batch-pooled, Dice epsilon 1 |
| `configs/bce_dice_posonly.yaml` | `bce_dice_pos` | BCE + 1.0 × soft Dice on slices with any positive target pixel |
| `configs/weighted_bce.yaml` | `weighted_bce` | BCE with `pos_weight` |
| `configs/bce_dice_w0.5.yaml` | `bce_dice` | BCE + 0.5 × soft Dice |
| `configs/bce_dice_w2.yaml` | `bce_dice` | BCE + 2.0 × soft Dice |

For `weighted_bce`, `pos_weight` is `auto`: foreground fraction `fg` is `foreground_pixels` summed over the train manifest divided by `n_train * 65536`, and the weight is `min((1 - fg) / fg, 100)`. The value used is stored as `pos_weight_used`.

`scripts/train.py` refuses to overwrite a directory that already contains `run_meta.json` unless `--max-train` is set. Do not pass `--max-train` or `--override` on the 14 study runs. `--max-train` sets `smoke_test: true`. `--override` on a full run appends `_ovr` to the config name and would no longer match the run directory names below.

Each finished run directory contains `best.pt`, `channel_stats.json`, `run_meta.json`, and `train_log.json`.

## Audit and validation reference

Set the data root, then audit before any training. The audit creates `evidence/` when it writes the default output.

```bash
export DATA=data

python scripts/audit_metadata.py --data-root "$DATA"
python scripts/eval_reference.py --data-root "$DATA"
```

`audit_metadata.py` writes `evidence/metadata_audit.json`. Confirm the patient and slice counts against the table above, patient overlaps of zero, dimensions all 256×256×3, and matching `patient_splits.csv` pair counts before training.

`eval_reference.py` scores an always-empty prediction on the **validation** masks only and writes `evidence/always_empty_val.json`. It does not accept a test split. The test-set always-empty reference is produced later, inside the single test evaluation, under the key `always_empty`.

## The 14 training runs

Four methods are trained at seeds 0, 1, and 2. The two Dice-weight sensitivities are seed 0 only. That is 12 + 2 = 14 runs. `bce_dice_w0.5` and `bce_dice_w2` were not trained at seeds 1 or 2.

```bash
python scripts/train.py --config configs/bce_baseline.yaml --data-root "$DATA" --seed 0
python scripts/train.py --config configs/bce_baseline.yaml --data-root "$DATA" --seed 1
python scripts/train.py --config configs/bce_baseline.yaml --data-root "$DATA" --seed 2

python scripts/train.py --config configs/bce_dice.yaml --data-root "$DATA" --seed 0
python scripts/train.py --config configs/bce_dice.yaml --data-root "$DATA" --seed 1
python scripts/train.py --config configs/bce_dice.yaml --data-root "$DATA" --seed 2

python scripts/train.py --config configs/bce_dice_posonly.yaml --data-root "$DATA" --seed 0
python scripts/train.py --config configs/bce_dice_posonly.yaml --data-root "$DATA" --seed 1
python scripts/train.py --config configs/bce_dice_posonly.yaml --data-root "$DATA" --seed 2

python scripts/train.py --config configs/weighted_bce.yaml --data-root "$DATA" --seed 0
python scripts/train.py --config configs/weighted_bce.yaml --data-root "$DATA" --seed 1
python scripts/train.py --config configs/weighted_bce.yaml --data-root "$DATA" --seed 2

python scripts/train.py --config configs/bce_dice_w0.5.yaml --data-root "$DATA" --seed 0
python scripts/train.py --config configs/bce_dice_w2.yaml --data-root "$DATA" --seed 0
```

Output directories:

```text
runs/bce_baseline_s0
runs/bce_baseline_s1
runs/bce_baseline_s2
runs/bce_dice_s0
runs/bce_dice_s1
runs/bce_dice_s2
runs/bce_dice_posonly_s0
runs/bce_dice_posonly_s1
runs/bce_dice_posonly_s2
runs/weighted_bce_s0
runs/weighted_bce_s1
runs/weighted_bce_s2
runs/bce_dice_w0.5_s0
runs/bce_dice_w2_s0
```

## Freeze

Freeze all 14 runs in one shot after method choices are final, and hash the three manifests. Freezing is refused if `FROZEN.json` already exists. Smoke runs are refused. The test manifest is hashed so its path and id columns cannot change afterward; hashing it does not score the test set.

Run paths and manifest paths are stored as absolute paths. Code hashes are relative to the repository root. Freeze and the test evaluation have to see the same absolute run directories and the same absolute manifest paths. Moving `data/` or `runs/` after freeze makes `verify_frozen` refuse the evaluation.

```bash
python scripts/freeze.py \
  --runs \
    runs/bce_baseline_s0 \
    runs/bce_baseline_s1 \
    runs/bce_baseline_s2 \
    runs/bce_dice_s0 \
    runs/bce_dice_s1 \
    runs/bce_dice_s2 \
    runs/bce_dice_posonly_s0 \
    runs/bce_dice_posonly_s1 \
    runs/bce_dice_posonly_s2 \
    runs/weighted_bce_s0 \
    runs/weighted_bce_s1 \
    runs/weighted_bce_s2 \
    runs/bce_dice_w0.5_s0 \
    runs/bce_dice_w2_s0 \
  --manifests \
    "$DATA/train_manifest.csv" \
    "$DATA/val_manifest.csv" \
    "$DATA/test_manifest.csv"
```

This writes `FROZEN.json` and `FROZEN.json.sha256`. The record holds, per run, the sha256 of `best.pt`, `channel_stats.json`, and `run_meta.json`, the selected threshold, the config, and the normalization stats. It also holds sha256 of `src/lggseg/data.py`, `src/lggseg/model.py`, `src/lggseg/metrics.py`, `src/lggseg/protocol.py`, and `scripts/evaluate_test.py`.

## Single test evaluation

Evaluate once, into a fresh output directory, after freeze. The completed study's evaluation artifacts (including `evidence/test/test_results.json`) live in the private companion, not here. A new run must not reuse a directory that already contains a completed `TEST_EVAL_LEDGER.json`, and it must not delete `TEST_EVAL.lock` or the ledger to obtain a second score. Deleting the lock does not reopen a completed evaluation.

```bash
export OUT=evidence/test_retrain
test ! -e "$OUT"

python scripts/evaluate_test.py \
  --data-root "$DATA" \
  --frozen FROZEN.json \
  --out-dir "$OUT" \
  --i-confirm-method-is-frozen
```

`$OUT` must be empty of any previous ledger. `evidence/test_retrain` is a fresh directory name so a replication does not write on top of an `evidence/test` tree from the 2026-11-08 evaluation. The flag `--i-confirm-method-is-frozen` is required.

What the script does, in order:

1. Re-checks `FROZEN.json` against `FROZEN.json.sha256`, the hashed source files, the manifest bytes, each run artifact, and the embedded normalization JSON.
2. Creates `$OUT` and claims it with `TEST_EVAL.lock` opened `O_EXCL`, then appends a `started` attempt to `$OUT/TEST_EVAL_LEDGER.json`, before any test image is read.
3. Pass 1 predicts from test images only. Mask paths are not opened. Normalization is the frozen train stats. The decision threshold is the frozen validation threshold.
4. The ledger is marked `test_labels_loaded: true`.
5. Pass 2 reads masks and scores. An always-empty prediction is scored on the same masks under the key `always_empty`.
6. On success it writes `$OUT/test_results.json` and sets the ledger attempt to `completed`.

Per-slice rows are written to `$OUT/<run-directory-name>_slices.csv`.

A failed attempt may be retried only with `--retry-reason`, and only when the ledger says test labels were not loaded. A failure after that mark, or a hard crash left at status `started`, is not a silent retry. The README states the same limit.

There is no supported second test evaluation of a frozen record in the same output directory.

## Metric definitions

All scores are at native 256×256. `slice_records` rejects any other shape.

Slice Dice and slice IoU use the overlap counts on that slice. With `tp`, `fp`, and `fn` the pixel counts:

- Dice = `2 * tp / (2 * tp + fp + fn)`
- IoU = `tp / (tp + fp + fn)`
- If the denominator is 0, the score is 1. Both the prediction and the ground truth are empty, so both-empty slice Dice is 1, and both-empty slice IoU is 1.
- If exactly one of the prediction and the ground truth is empty, the denominator is positive and the overlap is 0, so the slice score is 0.

Patient Dice is a volume Dice. For each `patient_id`, sum `tp`, `fp`, and `fn` over that patient's slices, then apply the Dice formula above, including the both-empty case (a patient with no lesion pixels and no false-positive pixels scores 1). `patient_dice_mean` is the unweighted mean of those per-patient volume Dice scores. Patients are not weighted by slice count. `patient_dice_median`, `patient_dice_std` (sample standard deviation, divisor `n - 1`, or 0 when only one patient is present), and `patient_dice_min` are reported beside it.

`positive_slice_dice` is the unweighted mean of slice Dice over slices whose ground truth is non-empty (`gt_empty == 0`). Both-empty slices score 1 and are included in `mean_slice_dice`; they are excluded from `positive_slice_dice`.

`empty_fp_rate` is the fraction of ground-truth-empty slices that have `fp > 0`. `empty_slices_with_fp` is that count. `empty_fp_pixels_total` is the sum of false-positive pixels on ground-truth-empty slices.

`pooled_foreground_dice` is one Dice from the sums of `tp`, `fp`, and `fn` over every slice in the split. That is the validation selection metric. It is not the headline test summary.

## Expected test results

`test_results.json` is a JSON object. The key `always_empty` is the all-zero prediction on the test masks. Every other key is the absolute path of a frozen run directory. Each value is the `summarize` object:

| Field | Definition |
| --- | --- |
| `n_slices`, `n_patients`, `n_positive`, `n_empty` | Counts on the scored split |
| `pooled_foreground_dice`, `pooled_foreground_iou` | One Dice / IoU from summed counts |
| `mean_slice_dice`, `mean_slice_iou` | Unweighted mean of per-slice scores, empties included |
| `positive_slice_dice`, `positive_slice_iou` | Unweighted mean on ground-truth-positive slices |
| `empty_slices_with_fp`, `empty_fp_rate` | Empty slices with any false positive, and that fraction |
| `empty_fp_pixels_total`, `empty_fp_pixels_mean` | False-positive pixel burden on empty slices |
| `patient_dice_mean`, `patient_dice_median`, `patient_dice_std`, `patient_dice_min` | Distribution of per-patient volume Dice |

Headline aggregation used for the four three-seed methods (`bce_baseline`, `bce_dice`, `bce_dice_posonly`, `weighted_bce`): the arithmetic mean and the standard deviation of `positive_slice_dice` across seeds 0, 1, and 2, together with `empty_fp_rate` for each method. `bce_dice_w0.5` and `bce_dice_w2` have seed 0 only, so they have a single positive-slice Dice and a single empty-slice FP rate, not a three-seed mean or standard deviation.

The 2026-11-08 numbers are omitted from the table. They were not available from this repository when this guide was written. `evidence/test/test_results.json`, the per-slice CSVs, the freeze record, the checkpoints, and the report are in the private companion repository, available on request.

| Method | Seeds | Positive-slice Dice | Empty-slice FP rate |
| --- | --- | --- | --- |
| always-empty reference | — | omitted | omitted |
| `bce_baseline` | 0, 1, 2 | omitted | omitted |
| `bce_dice` | 0, 1, 2 | omitted | omitted |
| `bce_dice_posonly` | 0, 1, 2 | omitted | omitted |
| `weighted_bce` | 0, 1, 2 | omitted | omitted |
| `bce_dice_w0.5` | 0 | omitted | omitted |
| `bce_dice_w2` | 0 | omitted | omitted |

## Nondeterminism

Seeds are applied. `seed_all` sets the Python `random` seed, the NumPy seed, and the PyTorch CPU and CUDA seeds. The training shuffle uses a `torch.Generator` seeded with the run seed. Each training sample's flip and rotation come from `np.random.default_rng([seed, epoch, index])`, which does not depend on worker count.

cuDNN benchmark and cuDNN deterministic mode are left at library defaults. The code does not call `torch.use_deterministic_algorithms` and does not set `cudnn.deterministic`. `run_meta.json` records whatever those flags were. DataLoader workers default to 4.

A retrain is therefore not expected to reproduce `best.pt` byte for byte, the checkpoint sha256, or `test_results.json` to the last digit. This repository does not define a numeric acceptance band. Matching the frozen hashes from the 2026-11-08 study is not a requirement for a retrain, and those hashes are not published here. A replication that changes the loss, the seed list, the threshold rule, the split, or the empty-slice convention is a different experiment.
