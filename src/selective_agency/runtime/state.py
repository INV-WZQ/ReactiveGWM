"""Model, Adam, schedule, progress, sampler and RNG at optimizer boundaries."""

import os
from pathlib import Path
import torch
import yaml

from .io import read_json, write_json
from .text import save_text_asset


class TrainingProgress:
    def __init__(self):
        self.values = {"global_step": 0, "samples_seen": 0}

    @property
    def step(self):
        return self.values["global_step"]

    @property
    def samples_seen(self):
        return self.values["samples_seen"]

    def advance(self, count):
        self.values["global_step"] += 1
        self.values["samples_seen"] += count
        if self.values.get("dataset_stage_samples_seen") is not None:
            self.values["dataset_stage_samples_seen"] += count

    def state_dict(self):
        return dict(self.values)

    def load_state_dict(self, values):
        self.values = {
            key: values[key]
            for key in (
                "global_step",
                "samples_seen",
                "dataset_stage_start_step",
                "dataset_stage_samples_seen",
            )
            if key in values
        }
        if int(self.step) < 0 or int(self.samples_seen) < 0:
            raise ValueError("invalid checkpoint progress")


def read_metadata(directory):
    path = Path(directory) / "metadata.json"
    return read_json(path) if path.is_file() else {}


def load_weights(model, directory):
    from selective_agency.models.loading import load_exported_checkpoint

    directory = Path(directory)
    if (directory / "pytorch_model.bin").is_file() and not list(directory.glob("*.safetensors")):
        model.load_state_dict(
            torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
        )
    else:
        load_exported_checkpoint(model, directory)


def runtime_topology(accelerator, config):
    settings = config["training"]
    return {
        "num_processes": accelerator.num_processes,
        "per_device_microbatch": settings["batch_size"],
        "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
    }


def restore(
    directory, mode, *, accelerator, model, optimizer, scheduler, progress, sampler, config
):
    directory = Path(directory).resolve()
    metadata = read_metadata(directory)
    if metadata.get("has_training_state") is False:
        raise ValueError("this checkpoint contains weights only; use --init-checkpoint instead of --resume")
    unwrapped = accelerator.unwrap_model(model)
    if (
        metadata.get("architecture_version", unwrapped.architecture_version)
        != unwrapped.architecture_version
    ):
        raise ValueError(
            "resume model architecture differs; use --init-checkpoint for a weights-only start"
        )
    if metadata.get("prompt_protocol", config["prompt_protocol"]) != config["prompt_protocol"]:
        raise ValueError("resume text protocol differs from the recipe")
    if metadata.get("method", config["method"]) != config["method"]:
        raise ValueError("resume method differs from the recipe")
    if mode == "exact":
        if (
            isinstance(metadata.get("precision"), str)
            and metadata["precision"] != config["training"]["precision"]
        ):
            raise ValueError(
                "precision changed; select --resume-mode reshard for a state conversion"
            )
        old = metadata.get("runtime_topology", {})
        new = runtime_topology(accelerator, config)
        if any(old.get(key, value) != value for key, value in new.items()):
            raise ValueError("worker/batch settings changed; select --resume-mode reshard")
        path = directory / f"sampler-rank-{accelerator.process_index:04d}.json"
        state = read_json(path)
        sampler.load_state_dict(state)
        accelerator.load_state(str(directory))
    else:
        load_weights(unwrapped, directory)
        optimizer.load_state_dict(
            torch.load(directory / "optimizer.bin", map_location="cpu", weights_only=False)
        )
        scheduler.load_state_dict(
            torch.load(directory / "scheduler.bin", map_location="cpu", weights_only=False)
        )
        progress.load_state_dict(
            torch.load(
                directory / "custom_checkpoint_0.pkl", map_location="cpu", weights_only=False
            )
        )
        old = read_json(directory / "sampler-rank-0000.json")
        if old["dataset_size"] != sampler.dataset_size:
            raise ValueError("resharding requires the same training records")
        # Use the saved cursor, not step * current batch: both final runs changed settings historically.
        old_total = (
            (old["dataset_size"] + old["num_replicas"] - 1)
            // old["num_replicas"]
            * old["num_replicas"]
        )
        consumed = old["epoch"] * old_total + old["cursor"] * old["num_replicas"]
        batch = (
            accelerator.num_processes
            * config["training"]["batch_size"]
            * config["training"]["gradient_accumulation_steps"]
        )
        details = sampler.seek_to_samples_seen(consumed, effective_global_batch=batch)
        from accelerate.utils import set_seed

        # Accelerate also seeds NumPy, which requires an unsigned 32-bit seed.
        resume_seed = (
            config["training"]["seed"] + progress.step * 1_000_003 + accelerator.process_index
        ) % (2**32)
        set_seed(resume_seed)
        accelerator.print(
            f"Resharded at step {progress.step}; replayed {details['replayed_samples']} samples."
        )
    accelerator.print(
        f"Restored step {progress.step}, samples_seen={progress.samples_seen} ({mode})."
    )


def save(output, *, accelerator, model, progress, sampler, config, text_tensors, text_metadata):
    output = Path(output)
    final = output / f"checkpoint-{progress.step}"
    temporary = output / f".checkpoint-{progress.step}.incomplete"
    if accelerator.is_main_process and (final.exists() or temporary.exists()):
        raise FileExistsError(f"checkpoint output already exists: {final}")
    accelerator.wait_for_everyone()
    # All ranks see the same path and participate in Accelerate's state save.
    accelerator.save_state(str(temporary), safe_serialization=True)
    write_json(
        temporary / f"sampler-rank-{accelerator.process_index:04d}.json", sampler.state_dict()
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        metadata = {
            "format": "selective_agency_training",
            **progress.state_dict(),
            "game": config["game"],
            "method": config["method"],
            "prompt_protocol": config["prompt_protocol"],
            "precision": config["training"]["precision"],
            "architecture_version": unwrapped.architecture_version,
            "runtime_topology": runtime_topology(accelerator, config),
            "loss": "future_latent_flow_matching_mse",
        }
        write_json(temporary / "metadata.json", metadata)
        saved_config = {
            **config,
            "paths": {
                key: os.path.relpath(value, final) for key, value in config.get("paths", {}).items()
            },
        }
        (temporary / "config.yaml").write_text(yaml.safe_dump(saved_config, sort_keys=False))
        save_text_asset(temporary / "text", text_tensors, text_metadata)
        os.replace(temporary, final)
    accelerator.wait_for_everyone()
    accelerator.print(f"Saved {final}")
    return final
