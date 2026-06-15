"""Training launcher with seamless resume support (ReactiveGWM-local).

Mirrors `diffsynth.diffusion.runner.launch_training_task` byte-for-byte,
adding three ReactiveGWM-local extensions:

  (1) After `accelerator.prepare(...)`, if `args.resume_state` is set,
      call `accelerator.load_state(args.resume_state)` — restores optimizer
      moments, LR scheduler state, RNG, and DataLoader sampler state
      (dataloader state restored only for prepared DataLoaders w/ workers
      below accelerate's checkpointable threshold). If the directory is named
      `state-<N>`, also resume `ModelLogger.num_steps` from N so checkpoint
      numbering and `--max_train_steps` continue globally.

  (2) At each save_steps boundary, if `args.save_full_state` is True, call
      `accelerator.save_state(<output>/state-<N>/)` ALONGSIDE the existing
      `step-<N>.safetensors` written by `ModelLogger.on_step_end`. The two
      products co-exist; `step-<N>.safetensors` remains the canonical
      lightweight (trainable-only) checkpoint, `state-<N>/` is the full
      Accelerate state directory used by `--resume_state`.

  (3) If `args.max_train_steps` is set, stop once `ModelLogger.num_steps`
      reaches that value. The count matches `--save_steps` and the
      `step-<N>.safetensors` checkpoint names.

No upstream `diffsynth/` code is touched. `train.py` dispatches into this
function via `launcher_map["sft"] = launch_training_task_with_state`.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from accelerate import Accelerator
from tqdm import tqdm

from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.runner import initialize_deepspeed_gradient_checkpointing
from diffsynth.diffusion.training_module import DiffusionTrainingModule


def _infer_step_from_state_path(path: str) -> Optional[int]:
    name = os.path.basename(os.path.normpath(path))
    prefix = "state-"
    if not name.startswith(prefix):
        return None
    value = name[len(prefix):]
    return int(value) if value.isdigit() else None


def launch_training_task_with_state(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: Optional[int] = None,
    num_epochs: int = 1,
    args=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    save_full_state = bool(args is not None and getattr(args, "save_full_state", False))
    resume_state_path = (args.resume_state if args is not None else None)
    max_train_steps = None if args is None else getattr(args, "max_train_steps", None)
    if max_train_steps is not None and max_train_steps <= 0:
        raise SystemExit("--max_train_steps must be a positive integer when set.")

    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers
    )
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    initialize_deepspeed_gradient_checkpointing(accelerator)

    # (1) Seamless resume: load full state AFTER prepare so accelerator
    #     knows about the prepared optimizer / scheduler / dataloader.
    if resume_state_path:
        if not os.path.isdir(resume_state_path):
            raise SystemExit(
                f"[Resume-State] --resume_state path is not a directory: {resume_state_path}"
            )
        accelerator.load_state(resume_state_path)
        resume_step = _infer_step_from_state_path(resume_state_path)
        if resume_step is not None:
            model_logger.num_steps = resume_step
        if accelerator.is_main_process:
            step_msg = (
                f"; logger step={resume_step}" if resume_step is not None else ""
            )
            print(
                f"[Resume-State] restored from {resume_state_path} "
                f"(optimizer + LR scheduler + RNG + dataloader{step_msg})",
                flush=True,
            )

    reached_max_train_steps = False
    trained_any_step = False
    if max_train_steps is not None and model_logger.num_steps >= max_train_steps:
        reached_max_train_steps = True
        if accelerator.is_main_process:
            print(
                f"[Max-Steps] current step {model_logger.num_steps} already "
                f">= target {max_train_steps}; skipping training.",
                flush=True,
            )
    for epoch_id in range(num_epochs):
        if reached_max_train_steps:
            break
        for data in tqdm(dataloader, disable=not accelerator.is_main_process):
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                trained_any_step = True

                # (2) Optionally dump full accelerator state alongside the
                #     step-<N>.safetensors written by model_logger above.
                if (
                    save_full_state
                    and save_steps is not None
                    and model_logger.num_steps % save_steps == 0
                ):
                    state_dir = os.path.join(
                        model_logger.output_path, f"state-{model_logger.num_steps}"
                    )
                    if accelerator.is_main_process:
                        os.makedirs(state_dir, exist_ok=True)
                    accelerator.wait_for_everyone()
                    accelerator.save_state(state_dir)
                    if accelerator.is_main_process:
                        print(f"[Save-State] wrote {state_dir}", flush=True)

                if (
                    max_train_steps is not None
                    and model_logger.num_steps >= max_train_steps
                ):
                    reached_max_train_steps = True
                    if accelerator.is_main_process:
                        print(
                            f"[Max-Steps] reached {model_logger.num_steps}; stopping.",
                            flush=True,
                        )
                    break

        if reached_max_train_steps:
            break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    if reached_max_train_steps and not trained_any_step:
        return
    if save_steps is None and reached_max_train_steps:
        model_logger.save_model(
            accelerator, model, f"step-{model_logger.num_steps}.safetensors"
        )
    else:
        model_logger.on_training_end(accelerator, model, save_steps)
