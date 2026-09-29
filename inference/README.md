# ReactiveGWM v2 Inference

Generate an HNM or Street Fighter III: New Generation clip from its initial image,
one mask per character, and per-character external actions or conditional NPC rules.
The CLI and public `ReactiveGWMPipeline` share the training model implementation.

## Install and base assets

From the repository root:

```bash
pip install -e .
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir checkpoints/base
```

The VAE decodes generated latents and encodes uncached first frames. The published
T5 tables cover the released actions/NPC rules; T5 weights are only needed for new
text. To encode new text:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir checkpoints/base
hf download Wan-AI/Wan2.1-T2V-1.3B --include 'google/umt5-xxl/*' \
  --local-dir checkpoints/tokenizer
```

## Run a released sample

```bash
python scripts/prepare_hf_dataset.py --game hnm --split val --limit 1 \
  --output data/hnm-example

python inference/inference.py \
  --checkpoint INV-WZQ/ReactiveGWM-v2-Models --model-subfolder HNM/main \
  --data data/hnm-example/samples.csv --split val --indices 0 \
  --vae checkpoints/base/Wan2.2_VAE.pth --gpus 0 --output outputs/hnm-example
```

For SF3, prepare `--game sf3` and select `--model-subfolder SF3/main`.
`--checkpoint` can instead point to a downloaded directory containing
`model.safetensors`, `config.yaml`, `metadata.json`, and `text/`.
The configuration defaults to the checkpoint's config; `--config` explicitly
overrides it. Game/architecture mismatches are rejected.

`--sample-id ID` selects by the published sample ID instead of row position.
`--indices 0 1` processes multiple rows in that split. Index order is the order
in the prepared CSV. Test samples can be selected with `--split test`; no future
video or future-video latent is read, even if present. Their first frame is
encoded on demand, so there is no requirement to build a test VAE cache first.

## Supply your own inputs or change character roles

```bash
python inference/inference.py \
  --checkpoint checkpoints/HNM/main \
  --image /path/to/first.png --masks /path/to/p1.png /path/to/p2.png \
  --controls /path/to/controls.json \
  --vae checkpoints/base/Wan2.2_VAE.pth --gpus 0 --output outputs/custom
```

Mask list order must match `controls.subjects`. Every subject supplies either
25 action names or one NPC prompt for the default 101-frame clip. Masks identify
physical characters; swap the action/prompt assignments to change which character
is externally controlled. Roles and rules remain fixed during one generation.
Use a separate generation to assign different roles.

The first frame must match the configured resolution (default 832 x 480). Masks
are reduced to soft patch coverage using the original model preprocessing. See
[examples](examples/README.md) for complete action lists and an input generator.

To override controls of an HF sample, pass `--data ... --controls controls.json`.
This uses the sample's image and masks with the new controls. New NPC sentences
or action names require `--text-encoder` and `--tokenizer`; new text is encoded
without truncation, and existing cached text rows retain their original values.
The source checkpoint/text table is never overwritten. Accepting a new sentence
does not guarantee that a model trained on a limited action/rule distribution
will execute it accurately.

## Python API

```python
from selective_agency import ReactiveGWMPipeline

pipe = ReactiveGWMPipeline.from_pretrained(
    "INV-WZQ/ReactiveGWM-v2-Models",
    subfolder="HNM/main",
    device="cuda",
    vae="checkpoints/base/Wan2.2_VAE.pth",
)
pipe.sample_csv(
    "data/hnm-example/samples.csv", split="val", index=0,
    output="outputs/api-sample", seed=20260808,
)
# Alternatively: pipe(image=..., masks=[...], controls={"subjects": [...]}, output=...)
```

## Parameters and output

| Argument | Default / purpose |
|---|---|
| `--steps` | 30 Wan flow-matching sampling steps |
| `--sigma-shift` | 5.0 |
| `--seed` | 20260808 |
| `--precision` | Checkpoint precision, normally BF16 |
| `--gpus` | Select visible physical GPU indices; inference uses one GPU |
| `--revision` | Pin the HF checkpoint revision; the resolved commit is recorded |
| `--hf-cache-dir` | Override model download cache |
| `--text-cache` | Explicit replacement T5 table for custom data |
| `--latent-only` | Skip VAE decoding; raw image inputs still need VAE encoding |

The released sampler uses CFG=1 and restores the clean first-frame latent after
every denoising step. Sampling is full-clip generation with causal self-attention;
this interface does not implement a streaming game engine.

CSV inference writes `sample-000000/` etc. under `--output`; custom input writes
directly into `--output`. Each contains `latents.safetensors`, `settings.json`,
and, unless `--latent-only` is used, `video.mp4`. Settings record the seed,
checkpoint source, geometry, actions, NPC prompts and sampling parameters.
Choose a new output directory for each run; existing sample outputs are not overwritten.
