"""ReactiveGWM unified training script (SF2 + SF3 + future game profiles).

Unified entrypoint for ReactiveGWM bidirectional training. The
per-game button schema, fixed-prompt fallback, and default video size are
sourced from `ReactiveGWM_Code.training.data.profiles`; pick one with `--game sf2` (default) or
`--game sf3`.

Supported modes (mutually compatible):

  A. Full DiT fine-tune        : --trainable_models dit
  B1. Scoped (keep cross_attn) : --trainable_models dit --trainable_filter cross_attn
  B2. Scoped (drop cross_attn) : --trainable_models dit --trainable_filter_exclude cross_attn
  C. LoRA on attention projections:
        --lora_base_model dit --lora_target_modules q,k,v,o --lora_rank 32
  + cache acceleration         : add --use_cached_dataset --cache_root <path>

Cold-start: the per-block ActionModule keys stay at their default (zero / xavier)
init; the rest of `ReactiveGWMModel` is shape-matched copied from Wan2.2-TI2V-5B.

Universal CFG-style dropouts (default 0.0; off-by-default for SF2 parity):
  --prompt_dropout_prob   per-step prob of replacing the prompt with empty string
  --action_dropout_prob   per-step prob of zeroing keyboard_action

Both add ZERO cost when set to 0 — bit-exact to the pre-merge SF2 trainer.
"""
import argparse
import os
import sys
import warnings
from pathlib import Path

import accelerate
import torch
from safetensors.torch import load_file

# Make the ReactiveGWM_Code package importable when this script is launched
# directly via `python training/bidirectional/train.py`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ReactiveGWM_Code.training.data.action_utils import get_action_op  # noqa: E402
from ReactiveGWM_Code.training.data.profiles import PROFILES, get_profile, profile_names  # noqa: E402
from ReactiveGWM_Code.training.data.prompt_utils import resolve_prompt  # noqa: E402

from diffsynth.core import UnifiedDataset  # noqa: E402
from diffsynth.diffusion import *  # noqa: E402,F401,F403
from diffsynth.models.reactive_gwm_dit import ReactiveGWMModel  # noqa: E402
from diffsynth.pipelines.reactive_gwm import (  # noqa: E402
    ModelConfig,
    ReactiveGWMPipeline,
)

# Local launcher that mirrors upstream launch_training_task and adds
# accelerator.save_state / load_state for seamless resume.
from ReactiveGWM_Code.training.bidirectional.runner_with_state import launch_training_task_with_state  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# When --use_cached_dataset is on, these 4 units are dropped from `pipe.units`
# and their outputs are injected from `data["__cached_*"]` in `forward()`.
# Names MUST match the classes in diffsynth/pipelines/reactive_gwm.py.
_CACHED_FILTERED_UNIT_NAMES = {
    "WanVideoUnit_PromptEmbedder",       # replaced by __cached_context_{posi,nega}
    "WanVideoUnit_InputVideoEmbedder",   # replaced by __cached_input_latents
    "WanVideoUnit_ImageEmbedderFused",   # replaced by __cached_first_frame_latents
    "WanVideoUnit_NoiseInitializer",     # manually reproduced (CUDA RNG parity)
}

WAN_MODEL_KWARGS = {
    "has_image_input": False,
    "patch_size": [1, 2, 2],
    "in_dim": 48, "dim": 3072, "ffn_dim": 14336, "freq_dim": 256,
    "text_dim": 4096, "out_dim": 48, "num_heads": 24, "num_layers": 30,
    "eps": 1e-06,
    "seperated_timestep": True,
    "require_clip_embedding": False,
    "require_vae_embedding": False,
    "fuse_vae_embedding_in_latents": True,
}


class FullModelLogger(ModelLogger):  # noqa: F405
    """Saves the entire trainable dit state dict (full or scoped param training)."""
    def __init__(self, output_path):
        super().__init__(output_path, remove_prefix_in_ckpt="pipe.dit.")


class ReactiveGWMTrainingModule(DiffusionTrainingModule):  # noqa: F405
    def __init__(
        self,
        profile,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        trainable_filter="",
        trainable_filter_exclude="",
        use_csv_prompt=True,
        prompt_column="prompt",
        use_cached_dataset=False,
        resume_from_ckpt=None,
        action_dropout_prob=0.0,
        prompt_dropout_prob=0.0,
    ):
        super().__init__()
        self.profile = profile
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing disabled — watch for OOM.")

        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths,
            fp8_models=fp8_models, offload_models=offload_models, device=device,
        )
        tokenizer_config = (
            ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B",
                        origin_file_pattern="google/umt5-xxl/")
            if tokenizer_path is None else ModelConfig(tokenizer_path)
        )
        self.pipe = ReactiveGWMPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=model_configs, tokenizer_config=tokenizer_config,
        )

        # Cold-start: build ReactiveGWMModel and copy shape-matched weights from
        # the pretrained Wan2.2-TI2V-5B DiT. ActionModule keys stay at their
        # default (zero / xavier) initialization. Optional resume_from_ckpt
        # then overlays a previously-saved DiT state dict (no optimizer state).
        custom_dit = ReactiveGWMModel(
            num_buttons=profile.num_buttons, **WAN_MODEL_KWARGS,
        ).to(torch.bfloat16)
        custom_dit = self._transfer_weights(custom_dit, self.pipe.dit)
        if resume_from_ckpt:
            state = load_file(resume_from_ckpt)
            missing, unexpected = custom_dit.load_state_dict(state, strict=False)
            print(f"[Resume] loaded {len(state)} keys from {resume_from_ckpt} "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
            del state
        self.pipe.dit = custom_dit.to(self.pipe.device)
        del custom_dit
        torch.cuda.empty_cache()

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        if trainable_models:
            for name in trainable_models.split(","):
                module = self.pipe.get_module(self.pipe, name)
                if module is not None:
                    module.train()
                    module.requires_grad_(True)

        self._apply_trainable_filter(trainable_filter, trainable_filter_exclude)

        self.use_cached_dataset = use_cached_dataset
        self.action_dropout_prob = action_dropout_prob
        self.prompt_dropout_prob = prompt_dropout_prob
        self._vae_z_dim = None
        self._vae_upsampling_factor = None
        if use_cached_dataset:
            # Stash VAE constants the dropped NoiseInitializer needs for the
            # CUDA-randn shape; we then drop pipe.vae / pipe.text_encoder.
            self._vae_z_dim = self.pipe.vae.model.z_dim
            self._vae_upsampling_factor = self.pipe.vae.upsampling_factor
            before = [type(u).__name__ for u in self.pipe.units]
            self.pipe.units = [u for u in self.pipe.units
                               if type(u).__name__ not in _CACHED_FILTERED_UNIT_NAMES]
            after = [type(u).__name__ for u in self.pipe.units]
            print(f"[cached] filtered {len(before) - len(after)} pipeline units")
            print(f"  before ({len(before)}): {before}")
            print(f"  after  ({len(after)}): {after}")
            self.pipe.vae = None
            self.pipe.text_encoder = None
            torch.cuda.empty_cache()
            print(f"[cached] dropped pipe.vae and pipe.text_encoder; "
                  f"z_dim={self._vae_z_dim}, upsampling_factor={self._vae_upsampling_factor}")

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.use_csv_prompt = use_csv_prompt
        self.prompt_column = prompt_column

    @staticmethod
    def _transfer_weights(custom_model, pretrained_dit):
        pretrained = pretrained_dit.state_dict()
        custom = custom_model.state_dict()
        new = {k: v for k, v in pretrained.items()
               if k in custom and v.shape == custom[k].shape}
        missing, _ = custom_model.load_state_dict(new, strict=False)
        print(f"[Cold-start] Transferred {len(new)}/{len(pretrained)} keys; "
              f"{len(missing)} keys remain at init (ActionModule).")
        return custom_model

    def _apply_trainable_filter(self, keep: str, drop: str):
        """Two complementary substring filters on dit.named_parameters().

        - keep: keep ONLY params whose name contains any of the comma-listed
                substrings (everything else frozen).
        - drop: freeze params whose name contains any of these substrings
                (everything else trainable).
        Both can be combined: keep is applied first, then drop further freezes.
        """
        if not (keep or drop):
            return
        keep_pats = [s.strip() for s in keep.split(",") if s.strip()]
        drop_pats = [s.strip() for s in drop.split(",") if s.strip()]
        for name, p in self.pipe.dit.named_parameters():
            if keep_pats and not any(pat in name for pat in keep_pats):
                p.requires_grad = False
                continue
            if drop_pats and any(pat in name for pat in drop_pats):
                p.requires_grad = False
        scalars = sum(p.numel() for p in self.pipe.dit.parameters() if p.requires_grad)
        tensors = sum(1 for p in self.pipe.dit.parameters() if p.requires_grad)
        print(f"[Trainable Filter] keep={keep_pats} drop={drop_pats} -> "
              f"{tensors} tensors trainable ({scalars:,} scalars)")

    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for x in extra_inputs:
            if x == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif x == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif x in ("reference_image", "vace_reference_image"):
                inputs_shared[x] = data[x][0]
            else:
                inputs_shared[x] = data[x]
        return inputs_shared

    def get_pipeline_inputs(self, data):
        prompt = resolve_prompt(data, self.profile, self.use_csv_prompt, self.prompt_column)
        inputs_posi = {"prompt": prompt}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "keyboard_action": data["action"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)

        # CPU RNG #1: action_dropout. With prob p, zero out the entire keyboard
        # so the model learns a "fix_prompt + no-action" baseline instead of
        # memorizing per-clip action-sequence hashes. No-op when prob == 0.
        if self.action_dropout_prob > 0.0 and torch.rand(1).item() < self.action_dropout_prob:
            inputs_shared = inputs[0]
            ka = inputs_shared.get("keyboard_action", None)
            if ka is not None:
                inputs_shared["keyboard_action"] = torch.zeros_like(ka)

        # CPU RNG #2: prompt_dropout. With prob p, swap to unconditional prompt.
        # Inference --cfg_scale>1 then extrapolates the delta to amplify NPC.
        drop_prompt = (
            self.prompt_dropout_prob > 0.0
            and torch.rand(1).item() < self.prompt_dropout_prob
        )

        if self.use_cached_dataset:
            inputs_shared, inputs_posi, inputs_nega = inputs
            # Inject cached T5 context (replaces dropped PromptEmbedder).
            cached_ctx = (
                data["__cached_context_nega"] if drop_prompt
                else data["__cached_context_posi"]
            )
            inputs_posi["context"] = cached_ctx.to(
                device=self.pipe.device, dtype=self.pipe.torch_dtype
            )
            inputs_posi.pop("prompt", None)
            # Inject cached VAE latents (replaces InputVideoEmbedder + ImageEmbedderFused).
            # ImageEmbedderFused side-effects we replicate: 'fuse_vae_embedding_in_latents'
            # and 'first_frame_latents' (loss reads both).
            inputs_shared["input_latents"] = data["__cached_input_latents"].to(
                device=self.pipe.device, dtype=self.pipe.torch_dtype
            )
            inputs_shared["first_frame_latents"] = data["__cached_first_frame_latents"].to(
                device=self.pipe.device, dtype=self.pipe.torch_dtype
            )
            inputs_shared["fuse_vae_embedding_in_latents"] = True
            # Placeholder; loss.py rebuilds via add_noise on input_latents.
            inputs_shared["latents"] = inputs_shared["input_latents"].clone()
            # CUDA RNG parity: NoiseInitializer (now dropped) consumed exactly one
            # randn call per step. Reproduce it so loss.py's downstream randn_like
            # sees the same RNG state as the non-cached path.
            _ = torch.randn(
                (
                    1,
                    self._vae_z_dim,
                    (inputs_shared["num_frames"] - 1) // 4 + 1,
                    inputs_shared["height"] // self._vae_upsampling_factor,
                    inputs_shared["width"] // self._vae_upsampling_factor,
                ),
                device=self.pipe.device,
                dtype=torch.float32,
            )
            inputs = (inputs_shared, inputs_posi, inputs_nega)
        else:
            # Non-cached path: empty the prompt string before PromptEmbedder runs.
            if drop_prompt:
                inputs_shared, inputs_posi, inputs_nega = inputs
                if "prompt" in inputs_posi:
                    inputs_posi["prompt"] = ""
                inputs = (inputs_shared, inputs_posi, inputs_nega)

        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, inputs_nega = inputs
        return FlowMatchSFTLoss(self.pipe, **inputs_shared, **inputs_posi)  # noqa: F405


def _bool_arg(s: str) -> bool:
    """argparse type for `--use_csv_prompt true|false` (case insensitive)."""
    return str(s).strip().lower() not in ("false", "0", "no", "off", "")


def build_parser():
    p = argparse.ArgumentParser(description="ReactiveGWM unified training (SF2/SF3)")
    p = add_general_config(p)  # noqa: F405
    p = add_video_size_config(p)  # noqa: F405
    p.add_argument("--game", type=str, default="sf2", choices=profile_names(),
                   help="Game profile selecting button schema, fixed prompt, "
                        "and 13-action eval grid. See training/data/profiles.py.")
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--max_timestep_boundary", type=float, default=1.0)
    p.add_argument("--min_timestep_boundary", type=float, default=0.0)
    p.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    p.add_argument("--action_hold_window", type=int, default=10,
                   help="Hold-last upsampling window for sparse parquet (10Hz->20fps). "
                        "Set 1 to disable.")
    # `--use_csv_prompt` accepts true/false explicitly; default is None so we
    # can fall back to the profile default after parsing (SF2=True, SF3=False).
    p.add_argument("--use_csv_prompt", type=_bool_arg, default=None,
                   help="Read per-clip prompt from CSV column. Default: profile-specific "
                        "(SF2=true, SF3=false). Pass `--use_csv_prompt true|false` to override.")
    p.add_argument("--prompt_column", type=str, default="prompt",
                   help="CSV column to read for the per-clip prompt.")
    p.add_argument("--trainable_filter", type=str, default="",
                   help="Comma-separated substrings; only DiT params whose name "
                        "contains ANY substring stay trainable. Empty = no filter.")
    p.add_argument("--trainable_filter_exclude", type=str, default="",
                   help="Comma-separated substrings; DiT params whose name "
                        "contains ANY substring are frozen. Empty = no filter.")
    p.add_argument("--use_cached_dataset", default=False, action="store_true",
                   help="Switch to CachedReactiveGWMDataset (pre-encoded VAE+T5).")
    p.add_argument("--cache_root", type=str, default=None,
                   help="Path to precomputed cache root (manifest.json + shards). "
                        "Required when --use_cached_dataset is set.")
    p.add_argument("--resume_from_ckpt", type=str, default=None,
                   help="Path to a previously saved DiT safetensors (output of "
                        "FullModelLogger). Loaded into ReactiveGWMModel after the "
                        "cold-start transfer_weights, before training starts. "
                        "Optimizer state is NOT resumed.")
    p.add_argument("--action_dropout_prob", type=float, default=0.0,
                   help="Per-step probability of zeroing keyboard_action (CFG-style).")
    p.add_argument("--prompt_dropout_prob", type=float, default=0.0,
                   help="Per-step probability of replacing the prompt with empty "
                        "string (CFG-style training). Default 0.0 (off).")
    p.add_argument("--save_full_state", default=False, action="store_true",
                   help="At each save_steps boundary, also dump "
                        "accelerator.save_state() to <output>/state-<step>/ "
                        "alongside the existing step-<step>.safetensors. "
                        "Enables seamless --resume_state.")
    p.add_argument("--max_train_steps", type=int, default=None,
                   help="Stop after this many training steps, counted the same "
                        "way as --save_steps and step-<N>.safetensors. If not "
                        "set, training runs for --num_epochs.")
    p.add_argument("--resume_state", type=str, default=None,
                   help="Path to a state-<N>/ directory written by "
                        "--save_full_state. Loaded via accelerator.load_state() "
                        "AFTER prepare() — restores optimizer moments, LR "
                        "scheduler, RNG, and DataLoader state. Mutually "
                        "exclusive with --resume_from_ckpt.")
    return p


def build_dataset(args, profile):
    if args.use_cached_dataset:
        if args.cache_root is None:
            raise ValueError("--use_cached_dataset requires --cache_root")
        from ReactiveGWM_Code.training.bidirectional.cached_dataset import CachedReactiveGWMDataset  # local import keeps non-cached runs lightweight
        return CachedReactiveGWMDataset(
            profile=profile,
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            cache_root=args.cache_root,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            action_hold_window=args.action_hold_window,
            use_csv_prompt=args.use_csv_prompt,
            prompt_column=args.prompt_column,
            repeat=args.dataset_repeat,
        )
    # Non-cached path. Only the action parquet operator is special; the rest
    # (video + first-frame image) come from UnifiedDataset's default video
    # operator. ReactiveGWM does not consume audio or animate streams.
    return UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height, width=args.width,
            height_division_factor=16, width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4, time_division_remainder=1,
        ),
        special_operator_map={
            "action": get_action_op(profile, args.dataset_base_path, args.num_frames,
                                    hold_window=args.action_hold_window),
        },
    )


if __name__ == "__main__":
    args = build_parser().parse_args()
    profile = get_profile(args.game)
    # Resolve profile-defaulted flags now that we know the game.
    if args.use_csv_prompt is None:
        args.use_csv_prompt = profile.default_use_csv_prompt
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise SystemExit("--max_train_steps must be a positive integer when set.")
    if args.resume_state and args.resume_from_ckpt:
        raise SystemExit(
            "--resume_state and --resume_from_ckpt are mutually exclusive: "
            "the former restores optimizer/LR/RNG/dataloader (seamless); the "
            "latter overlays a bare DiT safetensors (optimizer not restored)."
        )
    print(f"[ReactiveGWM] game={profile.name} ({profile.description})")
    print(f"[ReactiveGWM] use_csv_prompt={args.use_csv_prompt} prompt_column={args.prompt_column}")
    print(f"[ReactiveGWM] num_buttons={profile.num_buttons} button_cols={list(profile.button_cols)}")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(
            find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = build_dataset(args, profile)
    model = ReactiveGWMTrainingModule(
        profile=profile,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        trainable_filter=args.trainable_filter,
        trainable_filter_exclude=args.trainable_filter_exclude,
        use_csv_prompt=args.use_csv_prompt,
        prompt_column=args.prompt_column,
        use_cached_dataset=args.use_cached_dataset,
        resume_from_ckpt=args.resume_from_ckpt,
        action_dropout_prob=args.action_dropout_prob,
        prompt_dropout_prob=args.prompt_dropout_prob,
    )
    model_logger = FullModelLogger(args.output_path)
    # Training paths use the local launcher with seamless-resume hooks.
    # Data-process paths keep the upstream launcher unchanged.
    launcher_map = {
        "sft:data_process": launch_data_process_task,  # noqa: F405
        "direct_distill:data_process": launch_data_process_task,  # noqa: F405
        "sft": launch_training_task_with_state,
        "sft:train": launch_training_task_with_state,
        "direct_distill": launch_training_task_with_state,
        "direct_distill:train": launch_training_task_with_state,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
