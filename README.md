# LGG Brain MRI Binary Segmentation — BCE vs BCE+soft-Dice investigation

2D U-Net in PyTorch, trained from scratch, for binary tumor segmentation on lower-grade glioma (LGG) FLAIR MRI. The study compares plain BCE with BCE + soft Dice under a shared training budget, plus diagnostic and Dice-weight sensitivity runs.

**The dataset and trained checkpoints are not included.** Images, masks, manifest CSVs, `*.pt` / `*.pth` weights, `runs/`, freeze records, and `evidence/` stay on your machine and are gitignored. Checkpoints, evaluation evidence, and the written report are kept in a private companion repository and are available on request.

Status: **the full 14-run study was completed, frozen, and test-evaluated once on 2026-11-08.** Checkpoints and data are not included here. How to repeat training and the single test evaluation is in [REPLICATION.md](REPLICATION.md). Metrics computed only from this tree are smoke-test metrics unless a local `run_meta.json` says `"smoke_test": false`.

## Headline test metrics

The test headline is positive-slice Dice and the empty-slice false-positive rate. For `bce_baseline`, `bce_dice`, `bce_dice_posonly`, and `weighted_bce`, positive-slice Dice is the mean ± standard deviation over seeds 0, 1, and 2. `bce_dice_w0.5` and `bce_dice_w2` were trained at seed 0 only. Empty-slice FP rate is the fraction of ground-truth-empty slices with any false-positive pixel. Definitions are in [REPLICATION.md](REPLICATION.md).

Numeric cells are omitted. `evidence/test/test_results.json` is not in this repository, so these values were not computed here.

| Method | Seeds | Positive-slice Dice (mean ± SD) | Empty-slice FP rate |
| --- | --- | --- | --- |
| `bce_baseline` | 0, 1, 2 | omitted | omitted |
| `bce_dice` | 0, 1, 2 | omitted | omitted |
| `bce_dice_posonly` | 0, 1, 2 | omitted | omitted |
| `weighted_bce` | 0, 1, 2 | omitted | omitted |
| `bce_dice_w0.5` | 0 only | omitted | omitted |
| `bce_dice_w2` | 0 only | omitted | omitted |

## Setup

Python 3.10 or newer. CPU PyTorch is enough for the unit tests and a small smoke train. A full 50-epoch study is a GPU job (the study machine was an NVIDIA L40S).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On a SageMaker PyTorch GPU container, torch and CUDA come from the image. Install the extra packages from `requirements-gpu.txt` instead of pinning the CPU torch build in `requirements.txt`:

```bash
pip install -r requirements-gpu.txt
```

If you train on SageMaker, use a least-privilege execution role such as `<your-sagemaker-role>` that can read training and validation data only. Do not give that role the test holdout.

## Dataset

Download the public Kaggle dataset [`lgg-mri-segmentation`](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Mateusz Buda; FLAIR slices and segmentation masks from TCIA TCGA-LGG). The citation and the CC BY-NC-SA 4.0 terms are in Attribution below. This repository does not redistribute the images or masks.

The scripts do not read the raw Kaggle zip. Point `--data-root` at a prepared directory (for example `data/`, which is gitignored) that contains the manifests and the split trees:

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

`image_path` and `mask_path` are relative to that directory. Columns the code reads include `split`, `patient_id`, `slice_id`, `image_path`, `mask_path`, `height`, `width`, `channels`, `image_sha256`, `has_foreground`, and `foreground_pixels`. `patient_splits.csv` includes `patient_id` and `pair_count`. Images are 256×256×3 TIFFs. Masks are stored as 0/255 and binarized when loaded.

`test_manifest.csv` contains label-derived columns (`has_foreground`, `foreground_pixels`). Treat the test split as a holdout: do not inspect it while choosing the method. `scripts/train.py` loads the train and validation manifests only.

## Tests

Tests are synthetic. They do not need the Kaggle files or a checkpoint.

```bash
pytest -q tests
```

## Train

Files in `configs/` share one budget (50 epochs, batch size 16, AdamW, 32 base channels, the same validation selection rule). Only the loss changes. `configs/pilot_bce_3ep.yaml` is a 3-epoch throughput check, not a research result. The completed study trained `bce_baseline`, `bce_dice`, `bce_dice_posonly`, and `weighted_bce` at seeds 0, 1, and 2, and trained `bce_dice_w0.5` and `bce_dice_w2` at seed 0 only (14 runs). The command list, freeze, and the one-time test evaluation are in [REPLICATION.md](REPLICATION.md).

| Config | Loss |
| --- | --- |
| `configs/bce_baseline.yaml` | BCE (baseline) |
| `configs/bce_dice.yaml` | BCE + soft Dice, weight 1 |
| `configs/bce_dice_w0.5.yaml` | BCE + soft Dice, weight 0.5 |
| `configs/bce_dice_w2.yaml` | BCE + soft Dice, weight 2 |
| `configs/bce_dice_posonly.yaml` | BCE + soft Dice on positive slices only |
| `configs/weighted_bce.yaml` | BCE with a train-set `pos_weight` (capped) |
| `configs/pilot_bce_3ep.yaml` | 3-epoch BCE pilot |

```bash
export DATA=data

python scripts/audit_metadata.py --data-root "$DATA"
python scripts/eval_reference.py --data-root "$DATA"

# Twelve runs at seeds 0, 1, and 2. The checkpoint and the threshold are chosen on validation only.
# bce_dice_w0.5 and bce_dice_w2 are seed 0 only; they are not in this loop.
for c in bce_baseline bce_dice bce_dice_posonly weighted_bce; do
  for s in 0 1 2; do
    python scripts/train.py --config "configs/${c}.yaml" --data-root "$DATA" --seed "$s"
  done
done

# Dice-weight sensitivity, same budget, validation only: seed 0 only.
for c in bce_dice_w0.5 bce_dice_w2; do
  python scripts/train.py --config "configs/${c}.yaml" --data-root "$DATA" --seed 0
done
```

A finished run is written to `runs/<config-name>_s<seed>/` (`best.pt`, `channel_stats.json`, `run_meta.json`, `train_log.json`). Training refuses to overwrite a run that already has `run_meta.json`.

CPU smoke check (marked `"smoke_test": true`, and therefore not freezable):

```bash
python scripts/train.py --config configs/pilot_bce_3ep.yaml --data-root "$DATA" --max-train 32 --seed 0
```

## Freeze, evaluate, predict

Run this only after the method decisions are final. Freeze hashes each run's checkpoint, train-only normalization, and `run_meta.json`, plus the eval-path source and the manifest files you pass. Test evaluation then runs once.

```bash
python scripts/freeze.py \
  --runs runs/<chosen_run> \
  --manifests "$DATA/train_manifest.csv" "$DATA/val_manifest.csv" "$DATA/test_manifest.csv"

python scripts/evaluate_test.py --data-root "$DATA" --i-confirm-method-is-frozen
```

Pass every run you intend to score, not only one. `evaluate_test.py` writes under `evidence/test/` by default (gitignored), including `TEST_EVAL_LEDGER.json`. Use a fresh `--out-dir` when that directory already contains a ledger. Deleting `TEST_EVAL.lock` does not reopen a completed evaluation. The 14-run freeze and the one-time evaluation command are in [REPLICATION.md](REPLICATION.md).

Inference on individual TIFFs, using one run directory (checkpoint, channel stats, validation threshold):

```bash
python scripts/predict.py --run runs/<name>_s<seed> --out-dir outputs path/to/slice.tif
```

## Layout

- `src/lggseg/` data (manifest loader, tifffile channel-preserving read, binary masks, train-only channel stats, aligned flip/rot90 aug), `model.py` (2D U-Net from scratch, ~7.76M params at base=32), `losses.py`, `metrics.py`.
- `scripts/audit_metadata.py` split/hash/patient-disjoint checks; train/val label stats only.
- `scripts/eval_reference.py` always-empty reference on val.
- `scripts/train.py` trains one config; checkpoint and threshold chosen **on val only**; writes `run_meta.json`, `train_log.json`, `best.pt`, `channel_stats.json`.
- `scripts/freeze.py` one-shot freeze → `FROZEN.json` + `FROZEN.json.sha256`. Hashes per run `best.pt`, `channel_stats.json` (train-only normalization; also embedded verbatim), `run_meta.json` (threshold, config), and eval-path source (`src/lggseg/{data,model,metrics,protocol}.py`, `scripts/evaluate_test.py`). Refuses smoke runs and stats not `fitted_on: train`.
- `scripts/evaluate_test.py` (logic in `src/lggseg/protocol.py`): re-verifies every hash, then atomically claims the test set (`TEST_EVAL.lock`, O_EXCL) and an append-only `TEST_EVAL_LEDGER.json` **before** reading test data. Completed → never re-runs (deleting the lock alone does not reopen). Failure is logged; retry only with `--retry-reason` and only if it failed before test labels were loaded. Hard crash (status `started`) → refused, manual disclosure required.
- `scripts/predict.py` inference to 256×256 binary PNG masks.
- `configs/` identical budgets; only loss differs.
- `tests/` metric conventions (both empty=1, exactly one empty=0), alignment, shapes, losses.

## Test-protocol guarantees and limits

Evaluation is two-pass: PASS 1 predicts from test **images only** (no mask path opened); the ledger then records `test_labels_loaded=true`; PASS 2 reads masks and scores. Fix 2026-11-08 03:2x EST: earlier code used `SegDataset` (reads masks) during prediction, so a failure there could wrongly permit a "pre-label" retry. Regression tests instrument mask reads/file opens (fail on old code, pass now). Residual limit: the ledger marks label access at the start of PASS 2, conservatively — a failure after the mark but before any mask byte is read still blocks retry.

Guaranteed by code: tampering with frozen checkpoints, normalization stats, run metadata/thresholds, the freeze record, or hashed eval code is detected and evaluation refuses; a second evaluation via this code path is refused. **Not** guaranteed: someone can edit the code, delete the ledger/out dir, or score test predictions with another script. The ledger, freeze hashes, and an external experiment log are disclosed in the report as evidence; honesty of the experimenter remains the final safeguard.

## Metric conventions

All at native 256×256. Per-slice Dice/IoU: both empty = 1, exactly one empty = 0. Reported: pooled foreground Dice/IoU, mean slice Dice/IoU, positive-slice Dice/IoU, empty slices with any FP + FP pixel burden, patient-level Dice (pooled per patient) mean/median/std/min. Pixel accuracy not used.

## Attribution

- Data: Buda et al., Comput. Biol. Med. 109 (2019) 218–225; TCIA TCGA-LGG; Kaggle mateuszbuda/lgg-mri-segmentation; CC BY-NC-SA 4.0 (noncommercial; course redistribution reorganized files only).
- U-Net architecture: Ronneberger et al., MICCAI 2015 (implemented here from scratch; no pretrained weights).
- Soft Dice loss: Milletari et al., V-Net, 3DV 2016.
- Libraries: PyTorch, NumPy, pandas, tifffile.
- AI assistance: code scaffold, tests and proposal draft prepared with an AI assistant (Grok Bot); student must review, run and verify every result.

## Access-control note (2026-11-08)

- Training role `<your-sagemaker-role>`: least privilege, train/val only, no test access.
- Notebook role `<your-studio-role>`: a broad administrator-level policy was attached at the user's explicit request, and the test-data deny was removed. That environment can technically read `data/test_holdout/`. The protocol still forbids reading it until freeze and approval. Organization or resource policies may still restrict access.
