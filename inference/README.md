# ReactiveGWM Inference

Cleanroom inference package for **ReactiveGWM**. Loads a Wan2.2-TI2V-5B
backbone plus a ReactiveGWM SF2/SF3 DiT checkpoint, and synthesizes a
101-frame, 20 fps gameplay video conditioned on:

- a **first frame** (the starting screen state),
- a **button-stream parquet** (player actions, 10 Hz hold-upsampled to 20 fps),
- a **strategy-aware NPC prompt** (Offense / Control / Defense + behaviors).

The shipped CLI [`inference.py`](inference.py) is a thin wrapper around the
Python API in [`pipeline.py`](pipeline.py); both routes share the same defaults.

## Layout

```
inference/
├── README.md                ← you are here
├── __init__.py              ← public-API re-exports
├── inference.py             ← CLI entry-point (run from repo root)
├── pipeline.py              ← SFPipeline.from_pretrained / __call__
├── constants.py             ← variant defaults, button order, neg-prompt
├── models/
│   ├── __init__.py
│   ├── dit.py               ← WanModelAction (action-conditioned DiT)
│   ├── text_encoder.py      ← WanTextEncoder + WanTokenizer (UMT5)
│   └── vae.py               ← WanVideoVAE38
├── utils/
│   ├── __init__.py
│   ├── actions.py           ← parquet → action tensor, 10→20 fps upsample
│   ├── image.py             ← first-frame fit/crop, mp4 writer
│   └── scheduler.py         ← FlowMatchScheduler
└── examples/                ← per-strategy sample inputs (12 total)
    ├── SF2/{SF2.png, offense, control, defense}
    └── SF3/{SF3.png, offense, control, defense}
```

## Public Python API

```python
import torch
from PIL import Image
from ReactiveGWM_Code.inference import SFPipeline, NEG_PROMPT, VARIANT_DEFAULTS

pipe = SFPipeline.from_pretrained(
    base_model_dir="<base_model_dir>",
    checkpoint_path="<path-to-ckpt.safetensors>",
    variant="sf2",                       # or "sf3"
    torch_dtype=torch.bfloat16,
).to("cuda")

out = pipe(
    image=Image.open("first.png"),
    actions_parquet="actions.parquet",
    prompt=None,                         # None → variant default training prompt
    negative_prompt=NEG_PROMPT,
    num_frames=101,
    num_inference_steps=30,
    cfg_scale=5.0,
    action_cfg_scale=1.0,
    seed=2,
)
frames = out.frames[0]                   # list[PIL.Image], length = num_frames
```

`SFPipeline` is a `torch.nn.Module`, so `.to(device)` moves the DiT, VAE,
and T5 encoder together. `seed` is the *only* source of randomness;
fixing it makes the (prompt, actions) → video map deterministic.

## CLI: `inference.py`

```bash
# from the repo root
python inference/inference.py \
    --variant sf2 --ckpt <path-to-sf2.safetensors> \
    --image first.png --actions actions.parquet \
    --base_model <base_model_dir> --out out.mp4
```

### Arguments

| Flag | Default | Notes |
|---|---|---|
| `--variant` | *required* | `sf2` or `sf3`. Sets training prompt + native resolution (SF2: 480×608, SF3: 480×832). |
| `--ckpt` | *required* | ReactiveGWM DiT `.safetensors` from [`INV-WZQ/ReactiveGWM-Models`](https://huggingface.co/INV-WZQ/ReactiveGWM-Models). |
| `--image` | *required* | First-frame PNG/JPG. Aspect-preserving fit + center-crop to (H, W). |
| `--actions` | *required* | Parquet with 10 button columns (`UP DOWN LEFT RIGHT Y X Z A B C`). Sampled at 10 Hz, hold-upsampled to the 20 fps video grid. |
| `--base_model` | `/opt/dlami/.../base_model` | Directory holding Wan2.2 VAE + T5 weights + tokenizer. See layout below. |
| `--out` | `out.mp4` | Output 20 fps MP4. |
| `--num_frames` | `101` | Must satisfy `≡ 1 (mod 4)`; rounded up if not. |
| `--steps` | `30` | Denoising steps. |
| `--cfg` | `5.0` | Text-CFG (positive vs. negative prompt). |
| `--action_cfg` | `1.0` | Action-conditional CFG. Raise above 1 to amplify the action signal. |
| `--height`, `--width` | variant default | Rounded up to multiples of 32. |
| `--seed` | `2` | Locked-in default (chosen visually). Override for variation sweeps. |
| `--prompt` | variant default | Override the positive prompt. |
| `--negative_prompt` | `constants.NEG_PROMPT` | Shared Chinese neg-prompt baked in. |

### `--base_model` layout

```
<base_model_dir>/
└── Wan-AI/
    ├── Wan2.2-TI2V-5B/
    │   ├── Wan2.2_VAE.pth
    │   └── models_t5_umt5-xxl-enc-bf16.pth
    └── Wan2.1-T2V-1.3B/
        └── google/umt5-xxl/    # HF tokenizer files
```

## Strategy-aware NPC prompt

To steer NPC behavior, pass a prompt in the training-time format:

```
NPC: Active_Behavior(<a1>: <doc>; <a2>: <doc>; …),
     Passive_Behavior(<p1>: <doc>; <p2>: <doc>; …),
     Strategy(<class>: <doc>)
```

with `<class> ∈ {Offense, Control, Defense}`. The cross-attention strategy
modules ground this into NPC tactics; varying just the `Strategy(...)` slice
(with fixed first frame + actions) is the canonical way to demonstrate
strategy steering. Twelve ready-to-use prompts ship under `examples/`.

## Examples

Twelve per-strategy sample inputs (3 strategies × 2 samples × 2 variants),
drawn from [`INV-WZQ/ReactiveGWM-Datasets`](https://huggingface.co/datasets/INV-WZQ/ReactiveGWM-Datasets).
Each leaf `<NN>/` folder contains exactly two files:

| File | CLI flag | Notes |
|---|---|---|
| `prompt.txt` | `--prompt` | Strategy-aware NPC prompt with `Strategy(<class>: …)`. |
| `actions.parquet` | `--actions` | 100 rows × 10 buttons at 10 Hz. |

The **first frame is shared per variant**, sitting at the variant root
(`SF2/SF2.png`, `SF3/SF3.png`). Fixing it across all samples isolates each
rollout to the `(prompt, actions)` pair, which makes side-by-side strategy
comparison clean.

### Sample inventory (locked-in)

| Sample folder | SF2 source clip | SF3 source clip |
|---|---|---|
| `offense/01` | `clip_0000014` | `clip_0000104` |
| `offense/02` | `clip_0000020` | `clip_0000177` |
| `control/01` | `clip_0000008` | `clip_0000041` |
| `control/02` | `clip_0000044` | `clip_0000042` |
| `defense/01` | `clip_0000071` | `clip_0000098` |
| `defense/02` | `clip_0000074` | `clip_0000207` |

### Running an example

Assume the checkpoints and base assets have been laid out under `./models/`
and `./base_model/` (as per the top-level [README](../README.md#-setup)).
A complete run of `SF2/offense/01`:

```bash
python inference/inference.py \
    --variant    sf2 \
    --ckpt       ./models/SF2/ReactiveGWM_base.safetensors \
    --image      inference/examples/SF2/SF2.png \
    --actions    inference/examples/SF2/offense/01/actions.parquet \
    --prompt     "$(cat inference/examples/SF2/offense/01/prompt.txt)" \
    --base_model ./base_model \
    --out        out_sf2_offense_01.mp4
```

The prompt expands inline to:

```
NPC: Active_Behavior(Crouch: Enters and holds a crouching stance to lower
the character's hitbox and prepare charged moves.; Standing Punch:
Executes a basic punch attack while in a standing posture.),
Passive_Behavior(Idle: Stands perfectly still without any input, waiting
for an interaction.),
Strategy(Offense: Closes the distance quickly to apply pressure and
initiate close combat.)
```

Expected console output (H200 timing; pipeline load on cold cache ≈ 3.5 min,
warm-cache reload ≈ 1.5 min):

```
[SFPipeline] DiT loaded: 1145 keys (10 action_embedders), missing=…, unexpected=0
Denoising: 100%|██████████| 30/30 [00:16<00:00,  1.83it/s]
Saved → /…/out_sf2_offense_01.mp4
```

The output is a **608×480, 20 fps, 101-frame (≈ 5.05 s) MP4** seeded at 2.

To redirect NPC behavior with the same first frame + button stream, swap
the strategy folder:

```bash
# Same player inputs, defensive NPC instead of offensive:
--actions    inference/examples/SF2/defense/01/actions.parquet \
--prompt     "$(cat inference/examples/SF2/defense/01/prompt.txt)"
```

Swap `SF2`→`SF3` (and `--variant` / `--ckpt` / `--image`) to run the other
variant — SF3 outputs at 832×480 instead of 608×480.
