"""Encode or reuse the assets listed in a CSV, then write a prepared CSV."""

import argparse
from pathlib import Path
import shutil
from types import SimpleNamespace

from .cli import add_common_arguments, configured_path, configure_devices, torch_device
from .config import load_config, select_core
from .csv_data import CSVIndex, TENSOR_NAMES, write_dataset_csv
from .data import CachedDataset, validate_latents
from .geometry import VideoGeometry
from .text import import_text_asset, save_text_asset


def prepare(args, config):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    from .video import (
        _load_pipeline, _read_video, _encode_video, _encode_first_frame,
    )

    data = configured_path(args, config, "data", required=True)
    root = configured_path(args, config, "cache_root", required=True)
    output_csv = root / "samples.csv"
    if output_csv.exists():
        raise FileExistsError(f"choose a new cache directory: {output_csv} already exists")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    geometry = VideoGeometry(**config["video"])
    index = CSVIndex(
        data, text_cache=configured_path(args, config, "text_cache"),
        control_steps=geometry.control_steps,
    )
    splits = args.split or list(dict.fromkeys(row["split"] for row in index.records))
    if len(splits) != len(set(splits)):
        raise ValueError("choose distinct splits")
    selected = []
    for split in splits:
        rows = [row for row in index.records if row["split"] == split]
        if not rows:
            raise ValueError(f"CSV has no samples in split {split!r}")
        selected.extend(rows[:args.limit] if args.limit else rows)
    spec = index.text_spec()
    if index.text_path is not None and not args.encode_text:
        tensors, text = import_text_asset(index.text_path, spec)
    else:
        from .text import load_full_text_tokenizer, load_frozen_wan_text_encoder, encode_text_spec

        tokenizer_path = configured_path(args, config, "tokenizer", required=True)
        encoder_path = configured_path(args, config, "text_encoder", required=True)
        device = torch_device(args.device)
        tokenizer = load_full_text_tokenizer(tokenizer_path)
        encoder = load_frozen_wan_text_encoder(encoder_path, device)
        tensors, text = encode_text_spec(
            spec, tokenizer=tokenizer, text_encoder=encoder, device=device,
            batch_size=args.text_batch_size,
        )
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_text_asset(root, tensors, text)
    del tensors
    pipe = None
    counts = {}
    records = []
    for original in selected:
        row = dict(original)
        split = row["split"]
        split_number = splits.index(split)
        position = counts.get(split, 0)
        destination = root / "latents" / f"split-{split_number}" / f"{position:06d}.safetensors"
        names = row["tensor_names"]
        if row["latents"]:
            latent_path = Path(row["latents"])
            if not latent_path.is_file():
                raise FileNotFoundError(latent_path)
            if args.copy_assets:
                with safe_open(latent_path, framework="pt", device="cpu") as stored:
                    values = {
                        "input_latents": stored.get_tensor(names["video"]),
                        "first_frame_latents": stored.get_tensor(names["first"]),
                    }
                validate_latents(values, control_steps=geometry.control_steps, geometry=geometry, source=latent_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                save_file(values, str(destination))
                row["latents"], row["tensor_names"] = str(destination), dict(TENSOR_NAMES)
        else:
            if not row["video"]:
                raise ValueError(f"{row['record_id']}: provide latents or video")
            if pipe is None:
                vae = configured_path(args, config, "vae", required=True)
                pipe = _load_pipeline([str(vae)], None, str(torch_device(args.device)))
            record = SimpleNamespace(
                rgb_video=Path(row["video"]),
                x0_rgb=Path(row["first_frame"]) if row["first_frame"] else None,
            )
            frames = _read_video(record, geometry)
            if record.x0_rgb is None:
                record.x0_rgb = root / "first_frames" / f"split-{split_number}" / f"{position:06d}.png"
                record.x0_rgb.parent.mkdir(parents=True, exist_ok=True)
                frames[0].save(record.x0_rgb)
                row["first_frame"] = str(record.x0_rgb)
            with torch.inference_mode():
                values = {
                    "input_latents": _encode_video(pipe, frames)[0],
                    "first_frame_latents": _encode_first_frame(pipe, record.x0_rgb, geometry)[0],
                }
            validate_latents(values, control_steps=geometry.control_steps, geometry=geometry, source=row["video"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_file(values, str(destination))
            row["latents"], row["tensor_names"] = str(destination), dict(TENSOR_NAMES)
        if args.copy_assets:
            masks = []
            for slot, source in enumerate(row["x0_masks"]):
                mask = root / "masks" / f"split-{split_number}" / f"{position:06d}" / f"subject-{slot}.png"
                mask.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, mask)
                masks.append(str(mask))
            row["x0_masks"] = masks
        records.append(row)
        counts[split] = position + 1
        if (position + 1) % 1000 == 0:
            print(f"{split}: prepared {position + 1} records", flush=True)
    write_dataset_csv(output_csv, records, index.actions, text_cache=root / "control_text.safetensors")
    prepared = CSVIndex(output_csv, control_steps=geometry.control_steps)
    for split, count in counts.items():
        dataset = CachedDataset(prepared, split, geometry=geometry)
        for position in sorted({0, len(dataset) - 1}):
            dataset[position]
        print(f"{split}: prepared {count} records in {output_csv}", flush=True)


def main(default_config):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_config)
    parser.add_argument("--split", nargs="+", help="CSV splits; defaults to all splits")
    parser.add_argument("--encode-text", action="store_true", help="re-encode even when text_cache is supplied")
    parser.add_argument("--copy-assets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, help="prepare at most this many records per split")
    parser.add_argument("--vae")
    parser.add_argument("--text-encoder")
    parser.add_argument("--tokenizer")
    parser.add_argument("--text-batch-size", type=int, default=8)
    args = parser.parse_args()
    configure_devices(args)
    config = load_config(args.config)
    select_core(config["game"])
    prepare(args, config)
