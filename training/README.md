# ReactiveGWM v2 Training

This release trains the HNM and Street Fighter III: New Generation (SF3) main
models. The shared implementation is in `src/selective_agency/`; inference uses
the same model and conditioning code. The v1 bidirectional and Causal Forcing
workflows remain on the [v1 branch](https://github.com/INV-WZQ/ReactiveGWM/tree/v1).

## Install

Run commands from the repository root. Python 3.12 and a CUDA GPU are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python training/train.py --help
python training/prepare_cache.py --help
```

No separate DiffSynth checkout or directory named `ReactiveGWM_Code` is required.
The requirements pin the reference DiffSynth/PyTorch versions; `torchaudio` is
included because DiffSynth imports its data operators at startup.

## Prepare the released Hugging Face data

```bash
# Inspect archive sizes before downloading. By default, select two samples per split.
python scripts/prepare_hf_dataset.py --game hnm --split train val \
  --output data/hnm-smoke --dry-run
python scripts/prepare_hf_dataset.py --game hnm --split train val \
  --output data/hnm-smoke

# Full training/validation subset; this downloads hundreds of GB.
python scripts/prepare_hf_dataset.py --game hnm --split train val \
  --all --output data/hnm
python scripts/prepare_hf_dataset.py --game sf3 --split train val \
  --all --output data/sf3
```

Use `--sample-id ID ...` to select exact IDs, `--revision BRANCH_OR_TAG` to select
a readable version (default: `main`), or `--local-repo DIR` for already downloaded
HF files. `validation` is accepted as an alias for `val`. The output contains
`samples.csv`, extracted
`assets/`, downloaded archives in `downloads/`, and the branch/tag name and
archive inventory in `source.json`. Existing sample CSVs are not overwritten.

HF's `sample_index_*.csv` describes archive members; it is not itself a training
CSV. The helper preserves sample IDs, split membership, role/mask order and the
released control/text tables. It does not apply a second historical filter.

| Released subset | Train | Val | Test |
|---|---:|---:|---:|
| HNM | 20,000 | 187 | 192 |
| SF3 | 20,000 | 200 | 200 |

HNM's published splits omit the extra perfect-block strategy. The paper's table
lists 200 validation and 200 test clips; these commands use the current HF release.
Both games use 101 frames at 20 FPS and 832 x 480 resolution. There are 25 action
intervals, each aligned to four future frames, plus a clean first-frame anchor.

Train/val VAE and T5 caches are reused. Test has no VAE cache; add `--include-video`
if you intend to encode its complete target video. Ordinary inference only needs
its initial image, masks and controls, and encodes the first frame on demand.

## Train from Wan2.2

Download the three base DiT shards, or point `--base` to an existing copy:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --include 'diffusion_pytorch_model*.safetensors' --local-dir checkpoints/base

python training/train.py --config training/configs/hnm.yaml \
  --data data/hnm/samples.csv --base checkpoints/base \
  --gpus 0,1,2,3,4,5,6,7 --output outputs/hnm
```

For SF3, use `training/configs/sf3.yaml` and `data/sf3/samples.csv`. The internal
checkpoint game identifier remains `sf` to preserve compatibility with released
weights. All parameters of the DiT and role/control modules are trained except
the original frozen cross-attention output biases; VAE and T5 remain frozen.

## Fine-tune a released main model

```bash
python training/train.py --config training/configs/hnm.yaml \
  --data data/hnm/samples.csv \
  --init-checkpoint INV-WZQ/ReactiveGWM-v2-Models --model-subfolder HNM/main \
  --gpus 0 --max-steps 2000 --output outputs/hnm-finetune
```

`--init-checkpoint` also accepts a local model directory. For SF3, select
`SF3/main` and its training config. `--revision` and `--hf-cache-dir` control HF
initialization. HF exports contain model weights and text assets, **not optimizer
or sampler state**: initialization starts a new optimization run at step zero.

## Steps and batch size

Both main recipes default to **20,000 optimizer updates** and **effective batch 8**.
These defaults follow the paper; the supplied historical code used 28K/40K
recipes. This change does not imply the released weights were retrained at 20K.

| Argument | Default | Meaning |
|---|---:|---|
| `--max-steps` | 20000 | Total optimizer updates, including restored progress |
| `--effective-batch-size` | 8 | Target global batch when accumulation is automatic |
| `--batch-size` | 1 | Samples per GPU per microbatch |
| `--gradient-accumulation-steps` | auto | Explicit value overrides the target effective batch |
| `--learning-rate` | 5e-5 | AdamW learning rate |
| `--save-every` | 1000 | Save complete state every N updates; 0 saves only at the end |
| `--validate-every` | 1000 | Validation interval; 0 disables validation |
| `--num-workers` | 4 | DataLoader workers per process |
| `--precision` | bf16 | `bf16` or `fp32` |
| `--seed` | 20260808 | Initialization, sampling and role-permutation seed |

```text
effective batch = world size * batch size * gradient accumulation steps
```

Automatic accumulation is 8 on one GPU, 4 on two GPUs, and 1 on eight GPUs with
microbatch 1. The target must divide exactly; the program rejects incompatible
values. Explicit accumulation allows any positive resulting batch size. If
`--effective-batch-size` is supplied without explicit accumulation, accumulation
is recalculated even when the YAML contains a previous explicit value. Command-line
values override YAML values. The resolved settings are printed at startup.

`--gpus` launches one process per selected GPU; external `torchrun` is also
supported. `--dry-run` prints the resolved recipe without loading model weights.
There is no implicit video resize, crop or temporal resampling.

## Save and resume

```bash
python training/train.py --config training/configs/hnm.yaml \
  --data data/hnm/samples.csv --resume outputs/hnm/checkpoint-1000 \
  --gpus 0,1,2,3,4,5,6,7 --max-steps 20000 --output outputs/hnm-resumed
```

Complete checkpoints include weights, optimizer, LR scheduler, progress, sampler,
RNG, configuration and text tables. Exact resume requires the same data/order,
precision, world size and batch settings. `--resume-mode reshard` allows changing
world size/batch settings with the same training records; it reseeds RNG and reports
replayed samples. `--max-steps` is the total target, not additional updates after
resume. New output directories avoid checkpoint-name collisions. Validation is
automatically disabled if the CSV has no `val` split.

## Custom data and cache preparation

Each CSV row has the following columns. Paths are relative to the CSV; absolute
paths are also accepted.

| Column | Value |
|---|---|
| `sample_id`, `split` | Unique ID within the split; `train`, `val`, or `test` |
| `video` | Target RGB video, required if full video latents are absent |
| `first_frame` | Initial image; optional when preparing from the target video |
| `masks` | JSON list of mask paths in subject order |
| `controls` | JSON with `subjects`, each supplying `actions` or `prompt` |
| `latents` | Optional existing VAE safetensors |
| `text_cache` | Optional shared T5 safetensors, accompanied by its JSON metadata |

See [inference examples](../inference/examples/README.md) for controls. For a
101-frame video, each external subject supplies exactly 25 action names. NPC rules
are fixed within a rollout. Raw engine IDs and future NPC execution records are
not used as model inputs. An actually invisible subject may use `visible: false`
with an all-zero mask.

```bash
python training/prepare_cache.py --config training/configs/hnm.yaml \
  --data /path/to/custom.csv --cache-root data/custom-prepared \
  --vae checkpoints/base/Wan2.2_VAE.pth \
  --text-encoder checkpoints/base/models_t5_umt5-xxl-enc-bf16.pth \
  --tokenizer /path/to/umt5-tokenizer --gpus 0
```

Existing caches are reused; `--encode-text` explicitly rebuilds the text table.
`--no-copy-assets` references existing latent/mask files. Custom geometry is set
in YAML: width/height must be multiples of 32 and frames must be `4*T+1`.

## Validation

```bash
python tests/test_core.py
TRAINING_GAME=sf python tests/test_core.py
python tests/test_csv.py
TRAINING_GAME=sf python tests/test_csv.py
python tests/test_release.py
python tests/check_training.py --game hnm
python tests/check_training.py --game sf
```

The release's actual validation scope and results are recorded in
[VERIFICATION.md](../VERIFICATION.md).
