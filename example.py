"""End-to-end example. Run from inside this package directory:

    cd /path/to/ReactiveGWM_Code
    python example.py \\
        --variant sf3 --ckpt <path> --image first.png --actions actions.parquet --out out.mp4

Override the text prompt with --prompt (positive) and/or --negative_prompt;
both default to the variant's training-time prompt / the shared Chinese
neg-prompt baked into ``constants.py``.
"""
import argparse
import sys
from pathlib import Path

# Bootstrap: add this package's parent directory to sys.path so we can do
# `from ReactiveGWM_Code import ...` no matter where Python was invoked.
# Lets `python example.py ...` work directly without `-m`.
_PKG_DIR = Path(__file__).resolve().parent          # .../ReactiveGWM_Code
_PARENT = _PKG_DIR.parent                            # .../ReactiveGWM
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import torch
from PIL import Image

from ReactiveGWM_Code import NEG_PROMPT, SFPipeline, VARIANT_DEFAULTS
from ReactiveGWM_Code.utils import save_mp4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["sf2", "sf3"], required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True, help="First-frame image (png/jpg).")
    p.add_argument("--actions", required=True, help="Path to actions parquet.")
    p.add_argument("--out", default="out.mp4")
    p.add_argument("--base_model", default="/opt/dlami/nvme/zeqingwang/models/base_model")
    p.add_argument("--num_frames", type=int, default=101)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--action_cfg", type=float, default=1.0)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--prompt", type=str, default=None,
        help=f"Positive prompt. Default = variant default: "
             f"sf2={VARIANT_DEFAULTS['sf2']['prompt']!r}, "
             f"sf3={VARIANT_DEFAULTS['sf3']['prompt']!r}.",
    )
    p.add_argument(
        "--negative_prompt", type=str, default=NEG_PROMPT,
        help="Negative prompt for CFG. Default = the shared Chinese neg-prompt "
             "from ReactiveGWM.constants.NEG_PROMPT.",
    )
    args = p.parse_args()

    pipe = SFPipeline.from_pretrained(
        base_model_dir=args.base_model,
        checkpoint_path=args.ckpt,
        variant=args.variant,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    used_prompt = args.prompt if args.prompt is not None else VARIANT_DEFAULTS[args.variant]["prompt"]
    print(f"[example] prompt           = {used_prompt!r}")
    print(f"[example] negative_prompt  = {args.negative_prompt[:60]!r}...")

    out = pipe(
        image=Image.open(args.image),
        actions_parquet=args.actions,
        prompt=args.prompt,                    # None → pipeline picks variant default
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.steps,
        cfg_scale=args.cfg,
        action_cfg_scale=args.action_cfg,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )
    save_mp4(out.frames[0], args.out, fps=20)
    print(f"Saved → {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
