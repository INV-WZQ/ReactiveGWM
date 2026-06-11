"""Offline T5 + VAE precompute for ReactiveGWM training cache.

Walks the metadata CSV row-by-row, encodes each clip's video/first-frame via
the Wan2.2 VAE and each unique resolved prompt via T5, and writes sharded
.pt files under <cache_root>/{video,first_frame,t5}/<hash[:2]>/<hash>.pt.

Hash protocol MUST stay byte-identical with bidirectional/cached_dataset.py.

Usage (single GPU, full CSV):
  python -m ReactiveGWM_Code.training.bidirectional.precompute_cache \
    --game sf2 \
    --metadata_path <csv> --dataset_base_path <root> --cache_root <out> \
    --model_paths '[[...dit...], "<t5.pth>", "<vae.pth>"]' \
    --tokenizer_path <umt5_dir> \
    --height 480 --width 608 --num_frames 101 \
    --prompt_column prompt

Multi-GPU sharding:
  Run N copies with --shard_world_size=N --shard_rank=0..N-1, each on a
  different CUDA_VISIBLE_DEVICES. Atomic writes + --skip_existing make it safe.
  Pass --skip_manifest_write on non-rank-0 shards.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from diffsynth.core import UnifiedDataset  # noqa: E402
from diffsynth.pipelines.reactive_gwm import (  # noqa: E402
    ModelConfig,
    ReactiveGWMPipeline,
)

from ReactiveGWM_Code.training.data.profiles import get_profile, profile_names  # noqa: E402
from ReactiveGWM_Code.training.data.prompt_utils import resolve_prompt  # noqa: E402


CACHE_MANIFEST_VERSION = 1


# ---------- hashing (MUST match cached_dataset.py) ----------

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file_prefix(path: str, n_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(n_bytes))
    return h.hexdigest()[:16]


def video_cache_key(rel_path: str, H: int, W: int, nf: int, hdf: int, wdf: int,
                    first_frame: bool = False) -> str:
    base = f"{rel_path}|h={H}|w={W}|nf={nf}|hdf={hdf}|wdf={wdf}|fps=20"
    if first_frame:
        base += "|first_frame=1"
    return sha256_str(base)


def t5_cache_key(resolved_prompt: str) -> str:
    return sha256_str(f"t5|v1|{resolved_prompt}")


def _shard_path(root: Path, kind: str, h: str) -> Path:
    return root / kind / h[:2] / f"{h}.pt"


def atomic_save(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------- bit-exact encoders (mirror reactive_gwm.py units) ----------

def encode_prompt_bitexact(pipe: ReactiveGWMPipeline, prompt: str) -> torch.Tensor:
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        emb[:, v:] = 0
    return emb


def encode_video_bitexact(pipe: ReactiveGWMPipeline, pil_frames) -> torch.Tensor:
    input_video = pipe.preprocess_video(pil_frames)
    return pipe.vae.encode(input_video, device=pipe.device, tiled=False).to(
        dtype=pipe.torch_dtype, device=pipe.device,
    )


def encode_first_frame_bitexact(pipe: ReactiveGWMPipeline, first_pil,
                                height: int, width: int) -> torch.Tensor:
    image = pipe.preprocess_image(first_pil.resize((width, height))).transpose(0, 1)
    z = pipe.vae.encode([image], device=pipe.device, tiled=False)
    return z.to(dtype=pipe.torch_dtype, device=pipe.device)


# ---------- pipeline loading ----------

def load_pipeline(model_paths_json: str, tokenizer_path: str | None,
                  device: str, torch_dtype: torch.dtype) -> ReactiveGWMPipeline:
    paths = json.loads(model_paths_json)
    model_configs = [ModelConfig(path=p) for p in paths]
    tokenizer_config = (
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                    origin_file_pattern="google/umt5-xxl/")
        if tokenizer_path is None else ModelConfig(path=tokenizer_path)
    )
    pipe = ReactiveGWMPipeline.from_pretrained(
        torch_dtype=torch_dtype, device=device,
        model_configs=model_configs, tokenizer_config=tokenizer_config,
    )
    if hasattr(pipe, "dit") and pipe.dit is not None:
        pipe.dit = None
        torch.cuda.empty_cache()
    return pipe


# ---------- phases ----------

def phase_t5(pipe, df_full, profile, use_csv_prompt, prompt_column,
             cache_root, skip_existing):
    print("[phase 1/2] T5 encode")
    resolved = [resolve_prompt(df_full.iloc[i].to_dict(), profile,
                               use_csv_prompt, prompt_column)
                for i in range(len(df_full))]
    unique = sorted(set(resolved) | {""})
    print(f"  unique prompts: {len(unique)} (incl. empty)")
    table = {}
    t0 = time.time()
    for i, prompt in enumerate(unique):
        h = t5_cache_key(prompt)
        out = _shard_path(cache_root, "t5", h)
        table[h] = prompt[:60]
        if skip_existing and out.exists():
            continue
        with torch.no_grad():
            emb = encode_prompt_bitexact(pipe, prompt)
        atomic_save(out, emb.to(dtype=pipe.torch_dtype).cpu())
        if (i + 1) % 500 == 0 or (i + 1) == len(unique):
            print(f"  T5 {i+1}/{len(unique)} [{time.time()-t0:.1f}s]")
    return table


def phase_vae(pipe, df, profile, video_op, cache_root,
              height, width, num_frames, hdf, wdf,
              use_csv_prompt, prompt_column, skip_existing):
    print("[phase 2/2] VAE encode")
    rows_meta, failed = [], []
    t0 = time.time()
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        rel = row["video"]
        csv_index = int(row.get("_csv_index", i))
        vh = video_cache_key(rel, height, width, num_frames, hdf, wdf, False)
        ffh = video_cache_key(rel, height, width, num_frames, hdf, wdf, True)
        ph = t5_cache_key(resolve_prompt(row, profile, use_csv_prompt, prompt_column))
        rows_meta.append({"csv_index": csv_index, "rel_video": rel,
                          "video_hash": vh, "first_frame_hash": ffh, "prompt_hash": ph})
        v_path = _shard_path(cache_root, "video", vh)
        ff_path = _shard_path(cache_root, "first_frame", ffh)
        if skip_existing and v_path.exists() and ff_path.exists():
            continue
        try:
            pil_frames = video_op(rel)
            if not v_path.exists() or not skip_existing:
                with torch.no_grad():
                    latent = encode_video_bitexact(pipe, pil_frames)
                atomic_save(v_path, latent.to(dtype=pipe.torch_dtype).cpu())
            if not ff_path.exists() or not skip_existing:
                with torch.no_grad():
                    ff_latent = encode_first_frame_bitexact(pipe, pil_frames[0], height, width)
                atomic_save(ff_path, ff_latent.to(dtype=pipe.torch_dtype).cpu())
        except Exception as e:  # noqa: BLE001 — survive per-row failures
            failed.append({"csv_index": csv_index, "rel_video": rel, "error": repr(e)})
            print(f"  [fail] {rel}: {e}")
        if (i + 1) % 50 == 0 or (i + 1) == len(df):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-3)
            remain_min = (len(df) - i - 1) / max(rate, 1e-3) / 60
            print(f"  VAE {i+1}/{len(df)} [{elapsed:.0f}s, {rate:.2f} r/s, "
                  f"~{remain_min:.0f}m left, {len(failed)} failed]")
    return rows_meta, failed


def build_rows_meta(df_full, profile, height, width, num_frames, hdf, wdf,
                    use_csv_prompt, prompt_column):
    rows = []
    for i in range(len(df_full)):
        row = df_full.iloc[i].to_dict()
        rel = row["video"]
        csv_index = int(row.get("_csv_index", i))
        rows.append({
            "csv_index": csv_index,
            "rel_video": rel,
            "video_hash": video_cache_key(rel, height, width, num_frames, hdf, wdf, False),
            "first_frame_hash": video_cache_key(rel, height, width, num_frames, hdf, wdf, True),
            "prompt_hash": t5_cache_key(resolve_prompt(row, profile, use_csv_prompt, prompt_column)),
        })
    return rows


def _bool_arg(s: str) -> bool:
    return str(s).strip().lower() not in ("false", "0", "no", "off", "")


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(description="ReactiveGWM cache precompute (SF2/SF3)")
    p.add_argument("--game", default="sf2", choices=profile_names(),
                   help="Game profile (drives fixed_prompt fallback). "
                        "MUST match the --game value the trainer will use.")
    p.add_argument("--metadata", "--metadata_path", "--dataset_metadata_path",
                   dest="metadata", required=True, help="metadata.csv path")
    p.add_argument("--dataset_base", "--dataset_base_path",
                   dest="dataset_base", required=True,
                   help="Base directory CSV 'video' column is relative to")
    p.add_argument("--cache_root", required=True)
    p.add_argument("--height", type=int, default=None,
                   help="Defaults to profile.default_height.")
    p.add_argument("--width", type=int, default=None,
                   help="Defaults to profile.default_width.")
    p.add_argument("--num_frames", type=int, default=None,
                   help="Defaults to profile.default_num_frames.")
    p.add_argument("--height_division_factor", type=int, default=16)
    p.add_argument("--width_division_factor", type=int, default=16)
    p.add_argument("--max_pixels", type=int, default=1920 * 1080)
    p.add_argument("--model_paths", default=None,
                   help="JSON list matching train shell --model_paths "
                        "(DiT entries are loaded then dropped). Required unless --dry_run.")
    p.add_argument("--tokenizer_path", default=None)
    p.add_argument("--use_csv_prompt", default=None, type=_bool_arg,
                   help="Default: profile.default_use_csv_prompt.")
    p.add_argument("--prompt_column", default="prompt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=("bfloat16", "float16", "float32"))
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--shard_rank", type=int, default=0)
    p.add_argument("--shard_world_size", type=int, default=1)
    p.add_argument("--shard_weights", type=str, default=None,
                   help="Comma-separated positive ints; each shard rank gets that many "
                        "slots in a row-mod table. e.g. '1,1,1,1,2,2,2,2' → 8 shards "
                        "where ranks 4..7 each handle 2x as many rows as ranks 0..3. "
                        "Overrides --shard_world_size when set.")
    p.add_argument("--skip_manifest_write", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    profile = get_profile(args.game)
    if args.height is None: args.height = profile.default_height
    if args.width is None: args.width = profile.default_width
    if args.num_frames is None: args.num_frames = profile.default_num_frames
    if args.use_csv_prompt is None: args.use_csv_prompt = profile.default_use_csv_prompt
    print(f"[precompute] game={profile.name} ({profile.description})")
    print(f"[precompute] H={args.height} W={args.width} num_frames={args.num_frames} "
          f"use_csv_prompt={args.use_csv_prompt} prompt_column={args.prompt_column}")

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[args.torch_dtype]
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    print(f"[precompute] cache_root={cache_root}")

    df_full = pd.read_csv(args.metadata)
    print(f"[precompute] {len(df_full)} rows, columns={list(df_full.columns)}")
    if args.max_rows > 0 and args.max_rows < len(df_full):
        df_full = df_full.iloc[:args.max_rows].reset_index(drop=True)
        print(f"[precompute] truncated to {len(df_full)} rows (--max_rows)")
    df_full = df_full.assign(_csv_index=df_full.index.values)

    if args.shard_weights:
        weights = [int(x) for x in args.shard_weights.split(",") if x.strip()]
        if any(w <= 0 for w in weights):
            raise SystemExit("--shard_weights must be positive ints")
        args.shard_world_size = len(weights)
        if args.shard_rank >= args.shard_world_size:
            raise SystemExit(f"shard_rank {args.shard_rank} ≥ len(weights) {args.shard_world_size}")
        slots = []
        for r, w in enumerate(weights):
            slots.extend([r] * w)
        mask = df_full["_csv_index"].apply(lambda i: slots[i % len(slots)] == args.shard_rank)
        df = df_full[mask].reset_index(drop=True)
        print(f"[precompute] weighted shard {args.shard_rank} (w={weights[args.shard_rank]}/"
              f"{sum(weights)}, ranks={args.shard_world_size}): VAE on {len(df)} rows")
    elif args.shard_world_size > 1:
        mask = (df_full["_csv_index"] % args.shard_world_size) == args.shard_rank
        df = df_full[mask].reset_index(drop=True)
        print(f"[precompute] shard {args.shard_rank}/{args.shard_world_size}: "
              f"VAE on {len(df)} rows")
    else:
        df = df_full

    video_op = UnifiedDataset.default_video_operator(
        base_path=args.dataset_base,
        max_pixels=args.max_pixels,
        height=args.height, width=args.width,
        height_division_factor=args.height_division_factor,
        width_division_factor=args.width_division_factor,
        num_frames=args.num_frames,
        time_division_factor=4, time_division_remainder=1,
    )

    if args.dry_run:
        print("[precompute] DRY RUN: hashes only")
        rows_meta = build_rows_meta(
            df_full, profile, args.height, args.width, args.num_frames,
            args.height_division_factor, args.width_division_factor,
            args.use_csv_prompt, args.prompt_column,
        )
        print(f"[dry] first 3: {rows_meta[:3]}")
        return

    if args.model_paths is None:
        raise SystemExit("--model_paths required unless --dry_run")

    pipe = load_pipeline(args.model_paths, args.tokenizer_path, args.device, torch_dtype)
    vae_z_dim = pipe.vae.z_dim
    vae_up = pipe.vae.upsampling_factor

    if args.shard_rank == 0:
        prompt_table = phase_t5(pipe, df_full, profile, args.use_csv_prompt,
                                args.prompt_column, cache_root, args.skip_existing)
    else:
        print(f"[phase 1/2] shard {args.shard_rank} skips T5 (rank-0 only).")
        prompt_table = {}

    print("[precompute] unloading T5 before VAE phase")
    pipe.text_encoder = None
    torch.cuda.empty_cache()

    _, failed = phase_vae(
        pipe, df, profile, video_op, cache_root,
        args.height, args.width, args.num_frames,
        args.height_division_factor, args.width_division_factor,
        args.use_csv_prompt, args.prompt_column, args.skip_existing,
    )
    rows_meta = build_rows_meta(
        df_full, profile, args.height, args.width, args.num_frames,
        args.height_division_factor, args.width_division_factor,
        args.use_csv_prompt, args.prompt_column,
    )

    paths_list = json.loads(args.model_paths)
    vae_path = paths_list[-1] if isinstance(paths_list[-1], str) else str(paths_list[-1])
    t5_path = paths_list[-2] if len(paths_list) >= 2 and isinstance(paths_list[-2], str) else ""
    manifest = {
        "version": CACHE_MANIFEST_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv_path": str(Path(args.metadata).resolve()),
        "csv_md5": hashlib.md5(Path(args.metadata).read_bytes()).hexdigest(),
        "num_rows": int(len(df_full)),
        "dataset_base": str(Path(args.dataset_base).resolve()),
        "game": profile.name,
        "config": {
            "height": args.height, "width": args.width,
            "num_frames": args.num_frames,
            "height_division_factor": args.height_division_factor,
            "width_division_factor": args.width_division_factor,
            "max_pixels": args.max_pixels,
            "vae_upsampling_factor": vae_up,
            "vae_z_dim": vae_z_dim,
            "latent_shape": [1, vae_z_dim,
                             (args.num_frames - 1) // 4 + 1,
                             args.height // vae_up, args.width // vae_up],
            "first_frame_latent_shape": [1, vae_z_dim, 1,
                                         args.height // vae_up, args.width // vae_up],
            "cache_dtype": args.torch_dtype,
            "use_csv_prompt": bool(args.use_csv_prompt),
            "prompt_column": args.prompt_column,
            "fixed_prompt": profile.fixed_prompt,
        },
        "model_fingerprints": {
            "vae_path": vae_path,
            "vae_sha256_prefix": sha256_file_prefix(vae_path) if vae_path and Path(vae_path).exists() else "",
            "t5_path": t5_path,
            "t5_sha256_prefix": sha256_file_prefix(t5_path) if t5_path and Path(t5_path).exists() else "",
        },
        "rows": rows_meta,
        "prompt_table": prompt_table,
        "empty_prompt_hash": t5_cache_key(""),
        "failed_rows": failed,
    }
    if args.skip_manifest_write:
        print(f"[precompute] shard {args.shard_rank}: --skip_manifest_write")
    else:
        mp = cache_root / "manifest.json"
        tmp = mp.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        os.replace(tmp, mp)
        print(f"[precompute] manifest -> {mp}")
    if failed:
        print(f"[precompute] {len(failed)} rows failed (see manifest.failed_rows)")
    print("[precompute] DONE")


if __name__ == "__main__":
    main()
