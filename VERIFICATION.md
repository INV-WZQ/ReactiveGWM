# ReactiveGWM v2 release verification

Verified on 2026-09-29 using an isolated Python 3.12.3 environment on `Our-0`
(NVIDIA H200), with PyTorch 2.6.0/CUDA 12.4, DiffSynth 2.0.11 and the pinned
requirements. The installed package, repository entrypoints and shared model were
tested. Existing server environments, datasets and model files were not modified.

## Inputs

- Model repository: `INV-WZQ/ReactiveGWM-v2-Models`, commit
  `789c82a247f61a3bbb6eeec911ecd7dbae2f2e2f`; downloaded `HNM/main` and `SF3/main`.
- Dataset repository: `INV-WZQ/ReactiveGWM-v2-Datasets`, commit
  `ec39c556be6ce40ffabb26d906d423e2f79a26aa`.
- Numerical reference: the supplied `training_2026-09-29.zip`, extracted separately
  from the implementation under test.
- The existing Wan2.2 VAE, text encoder and tokenizer were used read-only.

Six real HF samples were prepared, one from each game/split:

| Game | Split | Sample ID |
|---|---|---|
| HNM | train | `hnm-train-production-13842-ca07f10504a899b27925a0c7-assigned` |
| HNM | val | `hnmr3-val-000199-main` |
| HNM | test | `hnmr3-test-000199-main` |
| SF3 | train | `sf3_train_external_000000` |
| SF3 | val | `sf3_v2_val_r11_000009` |
| SF3 | test | `sf3_v2_test_r11_000009` |

Archive selection, downloads, selective extraction, native test annotations and
portable CSV conversion all completed. Train/val tensors loaded as
`[48, 26, 30, 52]`, with matching 4096-dimensional T5 tables. The HNM validation
sample contains two NPCs; the SF3 validation sample contains one.

## Automated regression checks

| Check | Result |
|---|---|
| HNM core behavior | 6 passed |
| SF3 core behavior | 6 passed |
| HNM CSV and cache interfaces | 8 passed |
| SF3 CSV and cache interfaces | 8 passed |
| Release interfaces | 5 passed |
| HNM CPU/FP32 train-save-resume-infer CLI | Passed; continuous and resumed states exactly equal |
| SF3 CPU/FP32 train-save-resume-infer CLI | Passed; continuous and resumed states exactly equal |
| HNM two-process CPU/BF16 CLI | Passed; continuous and resumed states exactly equal |
| HNM numerical parity against supplied source | 151 forward/loss/gradient/update tensors exactly equal, CPU/FP32 |
| SF3 numerical parity against supplied source | 151 forward/loss/gradient/update tensors exactly equal, CPU/FP32 |

Core checks include causal visibility, excluding the clean anchor from the loss,
sampler state, reshard reseeding, and inference that still works after the future
video tensor is removed. Release checks cover automatic/explicit effective batch,
archive member selection and path/link rejection, uncached custom inputs, role
reassignment, and preserving existing text embeddings while adding new sentences.

The CLI resume checks compare weights, Adam state, LR scheduler, step/sample
counters and samplers after interrupted versus continuous runs. Their models are
small fixtures; the separate tests below use the actual 5B architecture.

## Actual 5B model tests

Both downloaded main checkpoints loaded strictly into their corresponding model.
Each generated its selected test sample with first-frame VAE encoding, 30 denoising
steps, sigma shift 5 and CFG 1. No future target frames or full-video cache were
required for inference.

A third HNM generation used the public Python API with explicit image/masks and
modified controls: an external character and NPC exchanged control roles, and a
new conditional sentence was encoded using the real frozen UMT5 encoder. Existing
cached sentences remained unchanged. This also completed 30-step generation.

All three MP4s were decoded and checked to contain 101 frames at 832 x 480 and
20 FPS. Generated `[48, 26, 30, 52]` latent tensors were finite. Frames 0, 33,
66 and 100 were visually inspected. These checks establish valid generation and
input handling, not a benchmark score or guaranteed compliance with a new rule.

Each main checkpoint also completed a real training update at the original
resolution and 101-frame length, using BF16, microbatch 1 and accumulation 8
(effective batch 8). The single selected training clip was repeated for this
functional check. Complete state was saved, then restored in a new process for
the next update:

| Game | Initial update loss | Restored update loss | Saved progress |
|---|---:|---:|---|
| HNM | 0.03812314 | 0.03633059 | steps 1 -> 2; samples 8 -> 16 |
| SF3 | 0.26439223 | 0.21693511 | steps 1 -> 2; samples 8 -> 16 |

Weights, optimizer, scheduler, progress and per-rank sampler files were present
in both games' checkpoints. This is short training and resume verification, not
a 20K-step retraining or reproduction of paper metrics.

The SF3 test clip also passed real cache preparation from its FFV1 video using
the Wan2.2 VAE, with its published T5 cache reused. The resulting cache was read
back through the ordinary training dataset loader.

Wan2.2 base initialization was checked separately on a full-size HNM model:
825 pretrained tensors (4,999,787,712 scalar values) loaded with no missing base
tensors. A full-resolution forward/backward pass completed with finite gradients
and loss 0.78138572. This verifies the base-initialization route in addition to
fine-tuning the released checkpoints.

Editable package installation and public API import passed. A CLI override check
resolved 321 steps, effective batch 12 and two GPUs to accumulation 6, confirming
that the 20K/batch-8 defaults are configurable.

## Reproduce the small checks

Run from the repository root after `pip install -e .`:

```bash
python tests/test_core.py
TRAINING_GAME=sf python tests/test_core.py
python tests/test_csv.py
TRAINING_GAME=sf python tests/test_csv.py
python tests/test_release.py
python tests/check_training.py --game hnm
python tests/check_training.py --game sf
python tests/check_training.py --game hnm --workers 2 --precision bf16
python tests/check_original.py --game hnm --original-training /path/to/original/training
python tests/check_original.py --game sf --original-training /path/to/original/training
```

Full-model inference, cache preparation and training commands are documented in
`inference/README.md` and `training/README.md`. Set `--max-steps 1`,
`--save-every 0`, `--validate-every 0` and `--num-workers 0` for the short GPU
training check; resume its `checkpoint-1` with `--max-steps 2` in a new output
directory.

## Cleanup

The task's isolated remote directory (about 154 GiB, including its environment,
downloaded HF assets, test checkpoints and generated outputs) was removed after
the logs, JSON reports and three sample videos were copied back locally. No test
process still used that directory at removal, and its absence was verified.
Existing server assets and environments were left intact. Local evidence is kept
under the ignored `.runtime/verification/` directory and is not part of the Git
release.
