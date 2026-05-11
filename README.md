# ReactiveGWM: Steering NPC in Reactive Game World Models

<p align="center">
  <img src="assets/SF2.gif" alt="SF2 demo">
</p>

> **ReactiveGWM: Steering NPC in Reactive Game World Models**   🥯[[Arxiv]]()   🌐[[Project Page]](https://reactivegwm.github.io/ReactiveGWM/)   
> [Zeqing Wang](https://inv-wzq.github.io/)<sup>12</sup>, [Danze Chen]()<sup>12</sup>, [Zhaohu Xing](https://ge-xing.github.io/)<sup>14</sup> , [Zizhao Tong]()<sup>15</sup> , [Yinhan Zhang]()<sup>16</sup> , [Xingyi Yang](https://adamdad.github.io/)<sup>3</sup> , [Yeying jin](https://jinyeying.github.io/)<sup>1</sup>    

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

## 🚀 Usage
```bash
# Street Fighter II: Champion Edition
python example.py \
    --variant sf2 --ckpt <path-to-sf2.safetensors> \
    --image first.png --actions actions.parquet \
    --base_model <base_model_dir> --out sf2.mp4

# Street Fighter Alpha 3
python example.py \
    --variant sf3 --ckpt <path-to-sf3.safetensors> \
    --image first.png --actions actions.parquet \
    --base_model <base_model_dir> --out sf3.mp4
```

Arguments:
- `--variant`: `sf2` or `sf3`. Selects the variant-specific training prompt and native resolution (`sf2` → 480×608, `sf3` → 480×832).
- `--ckpt`: ReactiveGWM DiT checkpoint (`.safetensors`) from [`INV-WZQ/ReactiveGWM-Models`](https://huggingface.co/INV-WZQ/ReactiveGWM-Models).
- `--image`: First-frame conditioning image (png/jpg). Aspect-preserving fit + center-crop, matching training-time pre-processing.
- `--actions`: Path to actions parquet with 10 button columns — `UP, DOWN, LEFT, RIGHT, Y, X, Z, A, B, C`. Sampled at 10Hz and hold-upsampled onto the 20fps video grid; sample parquet files are released in [`INV-WZQ/ReactiveGWM-Datasets`](https://huggingface.co/datasets/INV-WZQ/ReactiveGWM-Datasets).
- `--base_model`: Directory holding the Wan2.2 base assets (VAE, T5 weights, and tokenizer).
- `--num_frames`, `--steps`, `--cfg`, `--action_cfg`, `--height`, `--width`, `--seed`: Inference configuration. `--cfg` is text-CFG; `--action_cfg` is action-conditional CFG (raise above 1 to amplify the action signal). Defaults are `num_frames=101, steps=30, cfg=5.0, action_cfg=1.0, seed=0` and `H/W` from the variant default.
- `--prompt`, `--negative_prompt`: Override text prompts. Defaults are the variant-specific training prompt and the shared Chinese negative prompt in `constants.NEG_PROMPT`. To trigger strategy-aware NPC behavior, pass an NPC-style prompt that names a tactical category (Offense / Control / Defense) plus active/passive behaviors.

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
