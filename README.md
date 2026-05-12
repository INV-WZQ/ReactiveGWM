# ReactiveGWM: Steering NPC in Reactive Game World Models


<div align="center">
  <img src="assets/SF3.gif" width="70%" ></img>
  <br>
  <em>
      A reactive demo on the SF3 game.
  </em>
</div>
<br>

<a href="https://reactivegwm.github.io/ReactiveGWM/"><img src="https://img.shields.io/badge/Web-Project Page-1d72b8.svg" alt="Project Page"></a>
<a href=""><img src="https://img.shields.io/badge/arXiv-ReactiveGWM-A42C25.svg" alt="arXiv"></a>
<a href="https://huggingface.co/INV-WZQ/ReactiveGWM-Models"><img src="https://img.shields.io/badge/🤗_HuggingFace-Model-ffbd45.svg" alt="HuggingFace Model"></a>
<a href="https://huggingface.co/datasets/INV-WZQ/ReactiveGWM-Datasets"><img src="https://img.shields.io/badge/🤗_HuggingFace-Dataset-ffbd45.svg" alt="HuggingFace Dataset"></a>


> **ReactiveGWM: Steering NPC in Reactive Game World Models**    
> [Zeqing Wang](https://inv-wzq.github.io/)<sup>12</sup>, Danze Chen<sup>12</sup>, [Zhaohu Xing](https://ge-xing.github.io/)<sup>14</sup> , Zizhao Tong<sup>15</sup> , Yinhan Zhang<sup>16</sup> , [Xingyi Yang](https://adamdad.github.io/)<sup>3</sup> , [Yeying Jin](https://jinyeying.github.io/)<sup>1</sup>  
> <sup>1</sup> Tencent, <sup>2</sup> National University of Singapore, <sup>3</sup> The Hong Kong Polytechnic University  
> <sup>4</sup> The Hong Kong University of Science and Technology (Guangzhou), <sup>5</sup> University of Chinese Academy of Sciences, <sup>6</sup> The Hong Kong University of Science and Technology

## ⭐ Updates
- **[May 15, 2026]**: Inference Code, model and dataset are released.

## 📚 Introduction
**ReactiveGWM** is a novel game world model that synthesizes dynamic interactions between the player and **NPC**. Unlike current player-centric game world models, ReactiveGWM explicitly decouples player control from NPC autonomy: player actions are injected into the diffusion backbone via a lightweight additive bias, while high-level NPC strategies (Offense, Control, Defense) are grounded through cross-attention modules. These modules learn a *game-agnostic* representation of interactive logic, enabling zero-shot strategy transfer to vanilla world models of different games without retraining. Experiments on Street Fighter 2 and Street Fighter Alpha 3 show ReactiveGWM delivers fine-grained player controllability alongside autonomous, prompt-aligned NPC behavior.

<div align="center">
  <img src="assets/SF2.gif" width="70%" ></img>
  <br>
  <em>
      A reactive demo on the SF2 game.
  </em>
</div>
<br>

## 🛠️ Setup
```bash
conda create -n ReactiveGWM python=3.10
conda activate ReactiveGWM
pip install -r requirements.txt
```

Download the base model, the ReactiveGWM checkpoints, and (optionally) the strategy-aligned datasets:

| Resource | 🤗 Repo |
|---|---|
| Wan2.2 base (VAE + T5) | [`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) |
| UMT5 tokenizer | [`Wan-AI/Wan2.1-T2V-1.3B`](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) (`google/umt5-xxl/` subfolder) |
| ReactiveGWM checkpoints (SF2 / SF3) | [`INV-WZQ/ReactiveGWM-Models`](https://huggingface.co/INV-WZQ/ReactiveGWM-Models) |
| Strategy-aligned datasets | [`INV-WZQ/ReactiveGWM-Datasets`](https://huggingface.co/datasets/INV-WZQ/ReactiveGWM-Datasets) |

Arrange the base assets under a single `<base_model_dir>`:

```
<base_model_dir>/
└── Wan-AI/
    ├── Wan2.2-TI2V-5B/
    │   ├── Wan2.2_VAE.pth
    │   └── models_t5_umt5-xxl-enc-bf16.pth
    └── Wan2.1-T2V-1.3B/
        └── google/umt5-xxl/        # HF tokenizer files
```

## 🚀 Quick start

The repo ships **12 pre-built example samples** under
[`inference/examples/`](inference/examples) (3 strategies × 2 samples × 2
variants). Pick any one and run:

```bash
SAMPLE=inference/examples/SF2/offense/01

python inference/inference.py \
    --variant sf2 --ckpt <path-to-sf2.safetensors> \
    --image     inference/examples/SF2/SF2.png \
    --actions   $SAMPLE/actions.parquet \
    --prompt    "$(cat $SAMPLE/prompt.txt)" \
    --base_model <base_model_dir> \
    --out out.mp4
```

The first frame is shared per variant (`inference/examples/<variant>/<variant>.png`);
swap the `--image` along with `--variant` / `--ckpt` when moving from SF2 to
SF3. Swap `offense` / `control` / `defense` to redirect NPC behavior with the
same first frame and button stream.

For the full CLI argument table, the public Python API, module layout, and
the per-strategy example inventory, see [**`inference/README.md`**](inference/README.md).

## 🤓 Acknowledgments
ReactiveGWM is built on top of [Wan2.2-TI2V-5B](https://github.com/Wan-Video/Wan2.2) and adapts modeling / scheduler components from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). Gameplay recording uses the [stable-retro](https://github.com/Farama-Foundation/stable-retro) framework, and NPC strategy annotations are produced by [Gemini](https://deepmind.google/technologies/gemini/). We extend our gratitude to the open-source community for their valuable contributions!

## 🔗 Citation
```
```
