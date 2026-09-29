"""Exercise the real train/save/resume/infer CLI with a small CPU model."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
import yaml
from safetensors.torch import load_file

from fixtures import TRAINING, recipe, create_cache


def same(a, b, prefix="state"):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=lambda msg: f"{prefix}: {msg}")
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), prefix
        for key in a:
            same(a[key], b[key], prefix + "." + str(key))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), prefix
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, prefix + f"[{i}]")
    else:
        assert a == b, (prefix, a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", choices=("hnm", "sf"), default="hnm")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--frames", type=int, default=101)
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    from selective_agency.runtime.geometry import VideoGeometry

    control_steps = VideoGeometry(frames=args.frames).control_steps
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with tempfile.TemporaryDirectory(prefix="training-smoke-") as temporary:
        root = Path(temporary)
        cache = root / "cache"
        create_cache(
            cache, args.game, control_steps=control_steps,
            dtype=torch.float32 if args.precision == "fp32" else torch.bfloat16,
            include_validation=not args.train_only,
        )
        config = recipe(args.game, precision=args.precision, control_steps=control_steps)
        config["training"].update(
            num_workers=1, gradient_accumulation_steps=2, save_every=0,
            validate_every=1 if args.train_only else 0,
        )
        path = root / "small.yaml"
        path.write_text(yaml.safe_dump(config))
        entry = TRAINING / "training/train.py"
        command = [
            sys.executable,
            str(entry),
            "--config",
            str(path),
            "--data",
            str(cache / "samples.csv"),
            "--device",
            "cpu",
            "--num-processes",
            str(args.workers),
            "--log-every",
            "1",
        ]
        runs = [
            ("continuous", 3, ["--random-init"]),
            ("interrupted", 1, ["--random-init"]),
            ("resumed", 3, ["--resume", str(root / "interrupted/checkpoint-1")]),
        ]
        for name, steps, extra in runs:
            subprocess.run(
                [*command, "--output", str(root / name), "--max-steps", str(steps), *extra],
                check=True,
                env=env,
                cwd=root,
            )
        a, b = root / "continuous/checkpoint-3", root / "resumed/checkpoint-3"
        saved_config = yaml.safe_load((b / "config.yaml").read_text())
        assert (b / saved_config["paths"]["cache_root"]).resolve() == cache.resolve()
        assert (b / saved_config["paths"]["dataset"]).resolve() == (cache / "samples.csv").resolve()
        assert saved_config["training"]["validate_every"] == 0
        assert saved_config["video"]["frames"] == args.frames
        same(load_file(str(a / "model.safetensors")), load_file(str(b / "model.safetensors")))
        for name in ("optimizer.bin", "scheduler.bin", "custom_checkpoint_0.pkl"):
            same(
                torch.load(a / name, weights_only=False),
                torch.load(b / name, weights_only=False),
                name,
            )
        for rank in range(args.workers):
            name = f"sampler-rank-{rank:04d}.json"
            same(json.loads((a / name).read_text()), json.loads((b / name).read_text()), name)
        infer = TRAINING / "inference/inference.py"
        subprocess.run(
            [
                sys.executable,
                str(infer),
                "--config",
                str(path),
                "--checkpoint",
                str(b),
                "--data",
                str(cache / "samples.csv"),
                "--split",
                "train" if args.train_only else "val",
                "--output",
                str(root / "inference"),
                "--device",
                "cpu",
                "--steps",
                "2",
                "--latent-only",
            ],
            check=True,
            env=env,
            cwd=root,
        )
        generated = load_file(str(root / "inference/sample-000000/latents.safetensors"))["latents"]
        assert generated.shape[1] == control_steps + 1
        original = load_file(str(cache / ("train-0.safetensors" if args.train_only else "val-0.safetensors")))["first_frame_latents"].to(
            generated.dtype
        )
        same(generated[:, :1], original, "inference anchor")
        print(
            f"PASS {args.game} {args.precision}, {args.workers} workers, {args.frames} frames: continuous and resumed states exactly equal; inference anchor preserved"
        )


if __name__ == "__main__":
    main()
