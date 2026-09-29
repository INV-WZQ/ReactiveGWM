"""Shared path rules and a small single-node GPU/CPU launcher."""

import os
from pathlib import Path
import subprocess
import sys

from .io import resolve_path


def add_common_arguments(parser, default_config):
    parser.add_argument("--config", default=str(default_config), help="YAML recipe")
    parser.add_argument("--data", help="CSV sample index")
    parser.add_argument("--text-cache", help="override the CSV's shared T5 text table")
    parser.add_argument("--cache-root", help="prepared cache directory")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--gpus", help="visible physical GPUs, e.g. 0,2 (before CUDA initializes)")


def configured_path(args, config, name, *, required=False):
    value = getattr(args, name, None)
    if value is not None:
        return resolve_path(value, Path.cwd())
    value = config.get("paths", {}).get(name)
    if value is not None:
        return resolve_path(value, Path(args.config).resolve().parent)
    if required:
        raise ValueError(f"set --{name.replace('_', '-')} or paths.{name} in the recipe")
    return None


def dataset_path(args, config):
    if args.data is not None:
        return configured_path(args, config, "data", required=True)
    if args.cache_root is not None:
        return configured_path(args, config, "cache_root", required=True) / "samples.csv"
    prepared = configured_path(args, config, "dataset")
    if prepared is not None:
        return prepared
    data = configured_path(args, config, "data")
    if data is not None and (
        data.suffix.lower() == ".csv" or (data / "samples.csv").is_file()
    ):
        return data
    # Retained checkpoint recipes use paths.data for the original native source.
    return configured_path(args, config, "cache_root", required=True) / "samples.csv"


def configure_devices(args):
    if args.gpus is not None:
        values = args.gpus.split(",")
        if (
            not values
            or any(not v.strip().isdigit() for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("--gpus must be distinct comma-separated GPU indices")
        if args.device == "cpu":
            raise ValueError("--gpus cannot be combined with --device cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(v.strip() for v in values)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def launch_workers(args):
    configure_devices(args)
    count = args.num_processes or (len(args.gpus.split(",")) if args.gpus else 1)
    if count < 1 or (args.gpus and count > len(args.gpus.split(","))):
        raise ValueError("invalid number of workers for the selected GPUs")
    if args.device == "cpu":
        os.environ["ACCELERATE_USE_CPU"] = "true"
    if count == 1 or "RANK" in os.environ:
        return False
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={count}",
            str(Path(sys.argv[0]).resolve()),
            *sys.argv[1:],
        ],
        check=True,
    )
    return True


def torch_device(value):
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)
