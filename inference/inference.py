"""Generate from a prepared HF sample or a first frame, masks and controls JSON."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    from selective_agency.runtime.cli import configure_devices, torch_device

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="local directory or HF repo ID")
    parser.add_argument("--model-subfolder", choices=("HNM/main", "SF3/main"))
    parser.add_argument("--revision", help="HF model branch or tag (default: main)")
    parser.add_argument("--hf-cache-dir")
    parser.add_argument("--config", help="defaults to the checkpoint config.yaml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data", help="prepared samples.csv")
    source.add_argument("--image", help="custom first-frame image")
    parser.add_argument("--masks", nargs="+", help="one initial mask per subject, in controls order")
    parser.add_argument("--controls", help="controls JSON; replaces sample controls when --data is used")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--sample-id")
    selection.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--output", required=True)
    parser.add_argument("--vae")
    parser.add_argument("--text-cache")
    parser.add_argument("--text-encoder")
    parser.add_argument("--tokenizer")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--gpus")
    parser.add_argument("--precision", choices=("bf16", "fp32"))
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--sigma-shift", type=float)
    parser.add_argument("--latent-only", action="store_true")
    args = parser.parse_args()
    if args.image and (not args.masks or not args.controls):
        parser.error("--image requires --masks and --controls")
    if args.data and args.masks:
        parser.error("--masks is used with --image; CSV already specifies masks")
    if args.gpus and len(args.gpus.split(",")) != 1:
        parser.error("inference uses one GPU; select a single index with --gpus")
    if len(set(args.indices)) != len(args.indices):
        parser.error("choose distinct indices")
    configure_devices(args)
    from selective_agency.runtime.pipeline import ReactiveGWMPipeline

    pipe = ReactiveGWMPipeline.from_pretrained(
        args.checkpoint, subfolder=args.model_subfolder, revision=args.revision, cache_dir=args.hf_cache_dir,
        config=args.config, device=torch_device(args.device), precision=args.precision,
        vae=args.vae, text_encoder=args.text_encoder, tokenizer=args.tokenizer,
    )
    common = dict(seed=args.seed, steps=args.steps, sigma_shift=args.sigma_shift,
                  latent_only=args.latent_only, text_cache=args.text_cache)
    if args.image:
        pipe(image=args.image, masks=args.masks, controls=args.controls, output=args.output, **common)
    else:
        indices = [0] if args.sample_id else args.indices
        for index in indices:
            # Use ordinals rather than sample IDs as filesystem paths.
            destination = Path(args.output) / f"sample-{index:06d}"
            pipe.sample_csv(args.data, split=args.split, index=index, sample_id=args.sample_id,
                            controls=args.controls, output=destination, **common)


if __name__ == "__main__":
    main()
