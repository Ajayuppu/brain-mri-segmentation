# Protocol audit before study/holdout — 2026-11-08 (America/New_York)
Source of truth: /workspace/meta/ML_Project1.txt. Audited code: Studio snapshot 20261108T093351Z (extracted/).
DISCLOSURE CORRECTION (reported approximately05:01 EST): the audit executor read test_manifest.csv header AND first two rows, including label-derived has_foreground/foreground_pixels. This violated the pre-freeze no-inspection instruction. No test images/masks reportedly read; no further metadata inspection. The exposure cannot be undone, was reported not used for method selection, and must remain an explicit final-report limitation. This documentation worker has NOT opened that manifest. No AWS calls or new GPU compute by this documentation task.
The original 50-epoch baseline source was NOT modified. Its launch/running/completion is NOT independently verified here. All described code fixes are STAGED ONLY in /workspace/ml-project1_proposed_fixes, not verified applied in Studio; “fixed” below means staged implementation, not deployed status. The executor reports 28 staged synthetic tests pass, including regression additions; original25passing tests missed the augmentation bug.

## Assignment vs plan
- 50 epochs: NOT required by the assignment. It requires an initial baseline before the ~1-page proposal (4A) and comparable budgets (4D). 50 epochs is our chosen shared budget.
- Required and planned: always-empty reference + U-Net baseline + proposed (4C); main/diagnostic/stability (4D); Dice, IoU, positive-slice Dice, empty-slice FP count+pixels, patient-level, at 256x256, both-empty=1 (5) - implemented in metrics.py and unit-tested.
- Must record configs, seeds, hardware, params, training time (4D) - done in run_meta; CUDA/cuDNN/driver/code/manifest hashes added in fix.
- Overlays + quantitative subgroup/failure analysis (4E) - pilot has these; repeat on final runs.
- Seeds 3 for main/diagnostic is a stability check, not 3 research questions (4D last sentence).

## Findings
1. BUG (fixed, staged): augmentation RNG. SegDataset held one np.random Generator copied into each of 4 DataLoader workers and re-copied every epoch (non-persistent workers). Demonstrated: 64 samples got only 16 unique draws and the identical draw set every epoch. Not leakage and identical across conditions, but it reduces augmentation diversity and is a correctness error. Fix: per-sample generator keyed (seed, epoch, index), set_epoch() each epoch.
2. GAP (fixed): freeze did not hash manifests (test IDs/paths could change after freeze). freeze(--manifests) now hashes and verify_frozen re-checks them; evaluate_test refuses if the test manifest was not hashed at freeze time. Legacy note: earlier pilot freeze/protocol tests ran without manifest hashing.
3. RISK (fixed): checkpoint chosen each epoch at fixed threshold 0.5, threshold tuned only afterwards on a coarse 0.1 grid. Pilot showed a cliff between 0.6 and 0.7, and weighted_bce (pos_weight ~cap 100) shifts probabilities upward, so a fixed 0.5 biases epoch selection against it. Fix: each epoch scored at max over grid 0.05..0.95 (step 0.05); same rule for all conditions; still validation-only.
4. RISK (fixed): run dir overwrite. nb02 baseline cell writes runs/study/bce_baseline_s0, the same dir the old nb03 plan used. train.py now refuses to overwrite a completed run; study_v2 uses runs/study_v2.
5. Environment: torch 2.8.0, CUDA 12.9 (torch build), driver 595.112.07 reports CUDA 13.2, NVIDIA L40S, Python 3.12.15 (from nb00/run_meta). cudnn.benchmark/deterministic left default: runs are seeded but not bitwise reproducible; disclose. Fix records these per run.
6. Loss comparability: same model, data, optimizer, LR, epochs, batch size, augmentation and selection rule; BCE+Dice changes loss scale (gradient magnitude) - the Dice-weight 0.5/2 sensitivity runs address this. Soft Dice is batch-pooled (positive-only variant drops empty slices). Document.
7. weighted_bce pos_weight = (1-fg)/fg from TRAIN manifest, cap 100; value logged as pos_weight_used.
8. BatchNorm: batch 16, drop_last=True (no size-1 batches), eval() at validation - OK. Workers=4 now don't affect augmentation.
9. Metrics: patient score = pooled Dice per patient (patients with no lesion and no FP score 1) - state in report. Mean-slice Dice is inflated by empties; report positive-slice and empty-FP separately (done).
10. Freeze/one-shot: hashes checkpoints, norm stats, run_meta, eval code, FROZEN.json; O_EXCL lock + ledger; two-pass eval marks label access before first mask read. Cannot stop out-of-band misuse (documented). Holdout must not be staged until after freeze.
11. Holdout metadata: test_manifest.csv contains has_foreground/foreground_pixels (label-derived). Treat it as holdout: don't stage or analyse before freeze; hash only.

## Blocker?
No blocker for the running 50-ep baseline as a proposal-stage observation. Finding 1 is a blocker for the STUDY unless the user accepts disclosing it; recommended: apply fixes, re-run all 14 under study_v2 (bce_baseline_s0 included), keep the original source and any original50-epoch baseline artifacts EXACTLY in place as preliminary/proposal evidence, NOT matched comparative-study evidence. Do not rename/move them or claim relocation. All14comparative runs must be fresh corrected study_v2 runs after verified deployment/testing.

## Stage gates
G0 baseline: 50-ep BCE (current code) finishes -> save run_meta/log, backup.
G1 proposal: finalize ~1 page using baseline val numbers (pos-slice Dice, empty FP) -> docwriter.
G2 code freeze for study: apply staged fixes, 28 tests pass, record code hashes; no protocol change afterwards.
G3 study: 14 runs (manifest), val-only table, mean/SD over seeds, paired per-patient val differences.
G4 pre-registered final choice, written before holdout: freeze ALL 14 runs (+always-empty) with manifests; primary claim = bce_dice vs bce_baseline (3 seeds) on positive-slice Dice and empty-slice FP.
G5 holdout: stage test data, run evaluate_test.py once; any failure after label load is disclosed, not retried.
G6 report: 6-8 pages, overlays, subgroup (lesion size) + per-patient analysis, limitations, AI/assistance disclosure, decision-evidence table.

## Documentation crosslinks and current gates — approximately05:01 EST correction
- Proposal notes, technical record, chronological log, and report draft are private documents and are not published in this repository. Links and document IDs were removed.
- Patch: docs/fixes.diff (staged, not applied to Studio). Original pre-disclosure audit preserved in PROTOCOL_AUDIT_20261108.before_disclosure.md; its header-only wording is superseded.
- Rough14-run forecast2.7–3.8GPU-hours / USD7.6–10.6 at quoted StudioUSD2.80/hour, pilot-based estimate only; exclude idle/storage/requests and validate actual durations. App/kernels remain running by user instruction; do not shutdown.
- Fiftyepochs is chosen protocol, NOT assignment mandatory. Proposal review follows initialbaseline/dataaudit; no need to invent a mandatory50epoch baseline prerequisite.
