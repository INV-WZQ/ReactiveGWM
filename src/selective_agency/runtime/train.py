"""FM-only training with the original model, optimizer and update arithmetic."""

import argparse
import json
import os
from pathlib import Path
import time

from .cli import add_common_arguments, configured_path, dataset_path, launch_workers
from .config import load_config, select_core, make_model, bind_text, original_optimizer_config, resolve_batch_size
from .io import append_jsonl
from .geometry import VideoGeometry
from .batch import move_batch


class BatchStream:
    """Only successful microbatches advance the committed sampler cursor."""

    def __init__(self, dataset, sampler, *, batch_size, num_workers, seed):
        import torch

        self.dataset, self.sampler = dataset, sampler
        self.batch_size, self.num_workers = batch_size, num_workers
        self.generator = torch.Generator().manual_seed(seed)
        self.iterator = None

    def _restart(self):
        from torch.utils.data import DataLoader
        from selective_agency.data.collate import collate_samples

        self.iterator = iter(
            DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                sampler=self.sampler,
                num_workers=self.num_workers,
                collate_fn=collate_samples,
                pin_memory=True,
                generator=self.generator,
                persistent_workers=self.num_workers > 0,
                prefetch_factor=2 if self.num_workers > 0 else None,
            )
        )

    def next(self):
        if self.iterator is None:
            self._restart()
        try:
            return next(self.iterator)
        except StopIteration:
            self._restart()
            return next(self.iterator)



def validate(model, dataset, objective, accelerator, *, seed, limit=None):
    import torch
    from selective_agency.data.collate import collate_samples

    unwrapped = accelerator.unwrap_model(model)
    training = unwrapped.training
    unwrapped.eval()
    dtype = next(unwrapped.parameters()).dtype
    totals = torch.zeros(3, device=accelerator.device, dtype=torch.float64)
    count = min(len(dataset), limit) if limit else len(dataset)
    with torch.no_grad():
        for index in range(accelerator.process_index, count, accelerator.num_processes):
            batch = move_batch(collate_samples([dataset[index]]), accelerator.device, dtype)
            sample_seed = (seed + index * 1_000_003) % (2**63 - 1)
            cpu_generator = torch.Generator().manual_seed(sample_seed)
            noise_generator = torch.Generator(device=accelerator.device).manual_seed(sample_seed)
            timestep = int(
                torch.randint(objective.train_timesteps, (1,), generator=cpu_generator).item()
            )
            noise = torch.randn(
                batch["input_latents"].shape,
                dtype=dtype,
                device=accelerator.device,
                generator=noise_generator,
            )
            with accelerator.autocast():
                result = objective(
                    unwrapped,
                    batch,
                    use_gradient_checkpointing=False,
                    timestep_id=timestep,
                    noise=noise,
                )
            totals += torch.tensor(
                [result.loss.item(), result.unweighted_mse.item(), 1], device=totals.device
            )
    totals = accelerator.reduce(totals, reduction="sum")
    unwrapped.train(training)
    return {
        "validation_loss": (totals[0] / totals[2]).item(),
        "validation_mse": (totals[1] / totals[2]).item(),
        "validation_records": int(totals[2].item()),
    }


def run(args, config):
    import torch
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import set_seed
    from selective_agency.sampler import StatefulDistributedSampler
    from selective_agency.loss import WanFlowMatchingObjective
    from selective_agency.models.loading import load_shape_matched_wan
    from .data import CachedDataset
    from .optim import build_training_optimizer, build_training_scheduler
    from .state import TrainingProgress, restore, save, load_weights

    settings = config["training"]
    resolve_batch_size(settings, int(os.environ.get("WORLD_SIZE", "1")))
    for name in ("max_steps", "batch_size", "gradient_accumulation_steps", "train_timesteps"):
        if settings[name] < 1:
            raise ValueError(f"training.{name} must be positive")
    if min(settings["save_every"], settings["validate_every"], settings["num_workers"]) < 0:
        raise ValueError("save/validation intervals and workers must be nonnegative")
    if args.validation_samples is not None and args.validation_samples < 1:
        raise ValueError("--validation-samples must be positive")
    data = dataset_path(args, config)
    output = configured_path(args, config, "output", required=True)
    resolved_paths = {}
    for key in sorted(set(config.get("paths", {})) | {"cache_root", "output", "base", "text_cache"}):
        value = configured_path(args, config, key)
        if value is not None:
            resolved_paths[key] = str(value)
    config["paths"] = resolved_paths
    geometry = VideoGeometry(**config["video"])
    train = CachedDataset(
        data, "train", seed=settings["seed"], permutation="train_random",
        text_cache=configured_path(args, config, "text_cache"), geometry=geometry,
    )
    resolved_paths.update(dataset=str(train.index.path), cache_root=str(train.root))
    tensors, text = train.load_text(expected_dim=config["model"]["text_dim"])
    has_validation = any(row["split"] == "val" for row in train.index.records)
    skip_validation = bool(settings["validate_every"] and not has_validation)
    if skip_validation:
        settings["validate_every"] = 0
    validation = CachedDataset(train.index, "val", geometry=geometry) if settings["validate_every"] else None
    torch.use_deterministic_algorithms(settings["deterministic"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = settings["deterministic"]
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision="bf16" if settings["precision"] == "bf16" else "no",
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[
            DistributedDataParallelKwargs(
                find_unused_parameters=False, static_graph=True, bucket_cap_mb=25
            )
        ],
    )
    if skip_validation:
        accelerator.print("CSV has no val split; training validation is disabled.")
    if args.device == "cuda" and accelerator.device.type != "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")
    set_seed(settings["seed"], device_specific=False)
    model = make_model(config, accelerator.device, trainable=True)
    bind_text(model, tensors)
    if not args.resume and accelerator.is_main_process:
        if args.init_checkpoint:
            from .hub import resolve_checkpoint, validate_checkpoint

            checkpoint, _ = resolve_checkpoint(
                args.init_checkpoint, subfolder=args.model_subfolder,
                revision=args.revision, cache_dir=args.hf_cache_dir,
            )
            validate_checkpoint(checkpoint, config, model)
            load_weights(model, checkpoint)
        elif not args.random_init:
            base = configured_path(args, config, "base", required=True)
            shards = sorted(base.glob("diffusion_pytorch_model*.safetensors"))
            if not shards:
                shards = sorted((base / "transformer").glob("diffusion_pytorch_model*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"no Wan DiT shards in {base}")
            load_shape_matched_wan(model, shards)
    recipe = original_optimizer_config(config)
    optimizer = build_training_optimizer(model, recipe)
    scheduler = build_training_scheduler(optimizer, recipe)
    progress = TrainingProgress()
    accelerator.register_for_checkpointing(progress)
    sampler = StatefulDistributedSampler(
        len(train),
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=settings["seed"],
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if args.resume:
        restore(
            args.resume,
            args.resume_mode,
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            progress=progress,
            sampler=sampler,
            config=config,
        )
    stream = BatchStream(
        train,
        sampler,
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        seed=settings["seed"] + 2_000_003 + accelerator.process_index,
    )
    objective = WanFlowMatchingObjective(
        train_timesteps=settings["train_timesteps"], sigma_shift=settings["sigma_shift"]
    )
    dtype = next(model.parameters()).dtype
    output.mkdir(parents=True, exist_ok=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_saved = progress.step if args.resume else -1
    accelerator.print(
        f"{config['game']}/{config['method']}: {accelerator.num_processes} workers, "
        f"batch={settings['batch_size']}, accumulation={settings['gradient_accumulation_steps']}, "
        f"effective batch={settings['effective_batch_size']}, "
        f"target step={settings['max_steps']}"
    )
    while progress.step < settings["max_steps"]:
        started = time.monotonic()
        losses, mses, examples = 0.0, 0.0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for _ in range(settings["gradient_accumulation_steps"]):
            batch = move_batch(stream.next(), accelerator.device, dtype)
            timestep = int(torch.randint(objective.train_timesteps, (1,), device="cpu").item())
            noise = torch.randn_like(batch["input_latents"])
            with accelerator.autocast():
                result = objective(
                    model,
                    batch,
                    use_gradient_checkpointing=settings["gradient_checkpointing"],
                    timestep_id=timestep,
                    noise=noise,
                )
            # Accelerate divides once by the configured accumulation count.
            # Every microbatch synchronizes DDP gradients, as in the final runs.
            accelerator.backward(result.loss)
            count = len(batch["record_id"])
            sampler.mark_consumed(count)
            losses += result.loss.item()
            mses += result.unweighted_mse.item()
            examples += count
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        progress.advance(examples * accelerator.num_processes)
        means = accelerator.reduce(
            torch.tensor([losses, mses], device=accelerator.device), reduction="mean"
        )
        metrics = {
            "step": progress.step,
            "samples_seen": progress.samples_seen,
            "loss": means[0].item() / settings["gradient_accumulation_steps"],
            "mse": means[1].item() / settings["gradient_accumulation_steps"],
            "learning_rate": learning_rate,
            "global_batch": examples * accelerator.num_processes,
            "seconds": round(time.monotonic() - started, 3),
        }
        if validation is not None and (
            progress.step % settings["validate_every"] == 0
            or progress.step == settings["max_steps"]
        ):
            metrics.update(
                validate(
                    model,
                    validation,
                    objective,
                    accelerator,
                    seed=settings["seed"],
                    limit=args.validation_samples,
                )
            )
        if accelerator.is_main_process:
            append_jsonl(output / "metrics.jsonl", metrics)
            if (
                progress.step == 1
                or progress.step % args.log_every == 0
                or progress.step == settings["max_steps"]
            ):
                print(json.dumps(metrics), flush=True)
        if settings["save_every"] and progress.step % settings["save_every"] == 0:
            save(
                output,
                accelerator=accelerator,
                model=model,
                progress=progress,
                sampler=sampler,
                config=config,
                text_tensors=tensors,
                text_metadata=text,
            )
            last_saved = progress.step
    if progress.step != last_saved:
        save(
            output,
            accelerator=accelerator,
            model=model,
            progress=progress,
            sampler=sampler,
            config=config,
            text_tensors=tensors,
            text_metadata=text,
        )
    accelerator.end_training()


def main(default_config):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, default_config)
    parser.add_argument("--output")
    parser.add_argument("--base", help="original Wan DiT shard directory")
    initial = parser.add_mutually_exclusive_group()
    initial.add_argument(
        "--init-checkpoint", help="load model weights; start new Adam and progress"
    )
    initial.add_argument("--resume", help="restore a complete new or original Accelerate state")
    parser.add_argument("--model-subfolder", help="HNM/main or SF3/main for HF initialization")
    parser.add_argument("--revision", help="HF model branch or tag (default: main)")
    parser.add_argument("--hf-cache-dir")
    initial.add_argument(
        "--random-init",
        action="store_true",
        help="explicit random initialization for small-model checks",
    )
    parser.add_argument("--resume-mode", choices=("exact", "reshard"), default="exact")
    parser.add_argument(
        "--num-processes", type=int, help="single-node worker count; defaults to selected GPU count"
    )
    for name in (
        "max_steps",
        "save_every",
        "validate_every",
        "batch_size",
        "effective_batch_size",
        "gradient_accumulation_steps",
        "num_workers",
        "seed",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--precision", choices=("bf16", "fp32"))
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--validation-samples", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--dry-run", action="store_true", help="show the resolved recipe without loading models"
    )
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error("--log-every must be positive")
    if args.device not in {"auto", "cpu", "cuda"}:
        parser.error("training --device is auto/cpu/cuda; select physical devices with --gpus")
    config = load_config(args.config)
    if args.effective_batch_size is not None and args.gradient_accumulation_steps is None:
        config["training"]["gradient_accumulation_steps"] = None
    for key in config["training"]:
        value = getattr(args, key, None)
        if value is not None:
            config["training"][key] = value
    if args.dry_run:
        from .cli import configure_devices

        configure_devices(args)
        count = int(os.environ.get("WORLD_SIZE", args.num_processes or (len(args.gpus.split(",")) if args.gpus else 1)))
        resolve_batch_size(config["training"], count)
        paths = {key: str(configured_path(args, config, key)) for key in ("cache_root", "output", "base")}
        if args.data or args.cache_root or config.get("paths", {}).get("dataset"):
            paths["dataset"] = str(dataset_path(args, config))
        print(
            json.dumps(
                {
                    "recipe": config,
                    "paths": paths,
                    "gpus": args.gpus,
                    "num_processes": count,
                },
                indent=2,
            )
        )
        return
    if launch_workers(args):
        return
    select_core(config["game"])
    run(args, config)
