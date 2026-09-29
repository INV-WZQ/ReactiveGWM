# ReactiveGWM v2: Flexible Control and NPC Reactivity in Game World Models

[Project Page](https://inv-wzq.github.io/ReactiveGWM/) ·
[Hugging Face Models](https://huggingface.co/INV-WZQ/ReactiveGWM-v2-Models) ·
[Hugging Face Datasets](https://huggingface.co/datasets/INV-WZQ/ReactiveGWM-v2-Datasets) ·
[v1 Code and Paper](https://github.com/INV-WZQ/ReactiveGWM/tree/v1)

> [Zeqing Wang](https://inv-wzq.github.io/)<sup>12</sup>, Danze Chen<sup>12</sup>, [Zhaohu Xing](https://ge-xing.github.io/)<sup>4</sup> , Zizhao Tong<sup>15</sup> , Yinhan Zhang<sup>16</sup> , [Xingyi Yang](https://adamdad.github.io/)<sup>3</sup> , [Yeying Jin](https://jinyeying.github.io/)<sup>12</sup>  
> <sup>1</sup> Tencent, <sup>2</sup> National University of Singapore, <sup>3</sup> The Hong Kong Polytechnic University  
> <sup>4</sup> The Hong Kong University of Science and Technology (Guangzhou), <sup>5</sup> University of Chinese Academy of Sciences, <sup>6</sup> The Hong Kong University of Science and Technology

`main` contains the v2 **HNM and Street Fighter III: New Generation (SF3)** main
training and inference code. The original SF2 / Street Fighter Alpha 3 release,
including its bidirectional and Causal Forcing training, is preserved on `v1`.
The project website remains on the unchanged `page` branch.

## Introduction

ReactiveGWM assigns control roles to individual characters at rollout initialization.
Spatial Role Binding grounds learned handles in initial-frame instance masks.
Unified Agency Conditioning binds each character's external action sequence or
conditional NPC rule to the same handle. Causal self-attention lets characters
respond to the current and preceding video context.

A rollout takes an initial image, one mask per character, and per-character
controls. The default output is 101 frames at 20 FPS and 832 x 480 resolution,
with 25 external-action intervals. HNM supports two to six characters; SF3 models
two-fighter interactions. Roles and NPC rules are fixed within a rollout.

## Setup

Python 3.12 and a CUDA GPU are recommended. Install from this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The package includes the shared model and data adapters and pins the reference
DiffSynth dependency. No separate DiffSynth checkout is needed.

## Quick start

Prepare a small HF sample selection, download the VAE, then generate a clip:

```bash
python scripts/prepare_hf_dataset.py --game hnm --split val --limit 1 \
  --output data/hnm-example

hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir checkpoints/base

python inference/inference.py \
  --checkpoint INV-WZQ/ReactiveGWM-v2-Models --model-subfolder HNM/main \
  --data data/hnm-example/samples.csv --split val --indices 0 \
  --vae checkpoints/base/Wan2.2_VAE.pth --gpus 0 --output outputs/hnm-example
```

For SF3, prepare `--game sf3` and select `--model-subfolder SF3/main`.
Use `--sample-id` to choose a particular published sample. Dataset archives are
about 2 GiB each; `prepare_hf_dataset.py --dry-run` lists the selected archive
sizes before downloading. The helper produces the portable CSV used by both
training and inference. Complete dataset downloads are only selected with `--all`.

Inference also accepts your own initial image, ordered masks, and a JSON file
assigning actions or NPC prompts to each character. See
[inference/README.md](inference/README.md) for custom inputs, new text encoding,
role reassignment, local checkpoints, and the Python API.

## Training

Both game recipes default to **20,000 optimizer updates and effective batch 8**.
Steps, batch size, gradient accumulation, learning rate, devices, and output paths
are configurable through YAML or CLI arguments. Automatic accumulation adapts the
reference batch to the selected GPU count.

```bash
python scripts/prepare_hf_dataset.py --game hnm --split train val --all --output data/hnm

python training/train.py --config training/configs/hnm.yaml \
  --data data/hnm/samples.csv \
  --init-checkpoint INV-WZQ/ReactiveGWM-v2-Models --model-subfolder HNM/main \
  --gpus 0,1,2,3,4,5,6,7 --output outputs/hnm
```

HF weights initialize a new fine-tuning run. They do not contain complete optimizer
or sampler state. Training from Wan2.2 base shards and exact/full-state resume
from newly saved checkpoints are documented in [training/README.md](training/README.md).

The code follows the current HF data release: HNM train/val/test =
20,000/187/192 and SF3 = 20,000/200/200. HNM differs from the paper's 200/200
evaluation split counts because the published version excludes the extra
perfect-block strategy. Train/val caches are provided; test inference encodes only
the initial frame and does not read future ground-truth frames.

## Layout and verification

- `training/`: training/cache entrypoints and the two configurable recipes.
- `inference/`: CLI, public pipeline re-export and input examples.
- `src/selective_agency/`: shared v2 model, loss, data adapters and runtime.
- `scripts/prepare_hf_dataset.py`: HF archive-to-CSV preparation.
- `tests/`: model behavior, data interfaces, numerical parity and resume checks.

See [VERIFICATION.md](VERIFICATION.md) for actual test scope and results.

## Acknowledgments and model assets

ReactiveGWM builds on [Wan2.2](https://github.com/Wan-Video/Wan2.2) and
[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). Downloaded
models and data retain the licenses and third-party notices in their respective
Hugging Face repositories. The original release's acknowledgments and citation
remain available on the [v1 branch](https://github.com/INV-WZQ/ReactiveGWM/tree/v1).
