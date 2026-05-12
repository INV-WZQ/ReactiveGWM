# ReactiveGWM: Steering NPC in Reactive Game World Models

<p align="center">
  <img src="assets/SF2.gif" alt="SF2 demo">
</p>

> **ReactiveGWM: Steering NPC in Reactive Game World Models**   🥯[[Arxiv]]()   🌐[[Project Page]](https://reactivegwm.github.io/ReactiveGWM/)   
> [Zeqing Wang](https://inv-wzq.github.io/)<sup>12</sup>, Danze Chen<sup>12</sup>, [Zhaohu Xing](https://ge-xing.github.io/)<sup>14</sup> , Zizhao Tong<sup>15</sup> , Yinhan Zhang<sup>16</sup> , [Xingyi Yang](https://adamdad.github.io/)<sup>3</sup> , [Yeying Jin](https://jinyeying.github.io/)<sup>1</sup>    

> <sup>1</sup> Tencent, <sup>2</sup> National University of Singapore, <sup>3</sup> The Hong Kong Polytechnic University 

> <sup>4</sup> The Hong Kong University of Science and Technology (Guangzhou), <sup>5</sup> University of Chinese Academy of Sciences, <sup>6</sup> The Hong Kong University of Science and Technology

## 📚 Introduction
**ReactiveGWM** is a novel reactive game world model that synthesizes dynamic interactions between the player and NPC. Different from current game world models that simulate environments from a *player-centric* perspective — reducing the Non-Player Character (NPC) to background pixels — ReactiveGWM explicitly decouples player control from NPC autonomy: player actions are injected into the diffusion backbone via a lightweight additive bias, while high-level NPC strategies (Offense, Control, Defense) are grounded through cross-attention modules. These modules learn a *game-agnostic representation* of interactive logic, enabling **zero-shot strategy transfer** to off-the-shelf vanilla world models of different games without retraining. Experiments on *Street Fighter II* (SF2) and *Street Fighter Alpha 3* (SF3) show that ReactiveGWM delivers fine-grained player controllability alongside autonomous, prompt-aligned NPC behavior.

<div align="center">
  <img src="assets/SF3.gif" width="70%" ></img>
  <br>
  <em>
      A reactive rollout on the SF3 variant.
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
    --image     $SAMPLE/first_frame.png \
    --actions   $SAMPLE/actions.parquet \
    --prompt    "$(cat $SAMPLE/prompt.txt)" \
    --base_model <base_model_dir> \
    --out out.mp4
```

Swap `SF2`→`SF3` (and `--variant` / `--ckpt`) to run the other variant.
Swap `offense` / `control` / `defense` to redirect NPC behavior with the
same first frame and button stream.

For the full CLI argument table, the public Python API, module layout, and
the per-strategy example inventory, see [**`inference/README.md`**](inference/README.md).

## 🤓 Acknowledgments
ReactiveGWM is built on top of [Wan2.2-TI2V-5B](https://github.com/Wan-Video/Wan2.2) and adapts modeling / scheduler components from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). Gameplay recording uses the [stable-retro](https://github.com/Farama-Foundation/stable-retro) framework, and NPC strategy annotations are produced by [Gemini](https://deepmind.google/technologies/gemini/). We extend our gratitude to the open-source community for their valuable contributions!

## 🔗 Citation
```
@article{wang2026reactivegwm,
  title={ReactiveGWM: Steering NPC in Reactive Game World Models},
  author={Wang, Zeqing and Chen, Danze and Xing, Zhaohu and Tong, Zizhao and Zhang, Yinhan and Yang, Xingyi and Jin, Yeying},
  journal={arXiv preprint},
  year={2026}
}
```
