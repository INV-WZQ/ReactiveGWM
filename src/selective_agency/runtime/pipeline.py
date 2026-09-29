"""Public inference API for released samples and user-supplied character controls."""
import csv
import gc
import json
from pathlib import Path
import tempfile

from .config import load_config, make_model, bind_text
from .geometry import VideoGeometry
from .hub import resolve_checkpoint, validate_checkpoint
from .io import write_json


def extend_text(path, spec, *, encoder_path, tokenizer_path, device):
    """Encode new sentences and append them without changing existing T5 rows."""
    import torch
    from .text import load_text_asset, encode_text_spec, load_full_text_tokenizer, load_frozen_wan_text_encoder

    tensors, metadata = load_text_asset(path)
    old = metadata["spec"]
    if old["actions"] == spec["actions"] and old["npc_prompts"] == spec["npc_prompts"]:
        return tensors, metadata
    if not encoder_path or not tokenizer_path:
        raise ValueError("controls contain new text; provide --text-encoder and --tokenizer to encode it")
    added_actions = spec["actions"][len(old["actions"]):]
    added_prompts = spec["npc_prompts"][len(old["npc_prompts"]):]
    # The encoder's schema requires an Action row; the dummy is discarded below.
    encode_actions = added_actions or old["actions"][:1]
    new_spec = {
        "actions": encode_actions, "action_texts": [f"ACTION={x}" for x in encode_actions],
        "npc_prompts": added_prompts,
    }
    encoder = load_frozen_wan_text_encoder(encoder_path, device)
    new, details = encode_text_spec(
        new_spec, tokenizer=load_full_text_tokenizer(tokenizer_path), text_encoder=encoder, device=device,
    )
    del encoder
    gc.collect()
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    for prefix, count in (("action", len(added_actions)), ("npc", len(added_prompts))):
        if not count:
            continue
        hkey, mkey = prefix + "_hidden_states", prefix + "_attention_mask"
        width = max(tensors[hkey].shape[1], new[hkey].shape[1])
        hidden = tensors[hkey].new_zeros((tensors[hkey].shape[0] + count, width, tensors[hkey].shape[2]))
        mask = tensors[mkey].new_zeros(hidden.shape[:2])
        old_count = tensors[hkey].shape[0]
        hidden[:old_count, :tensors[hkey].shape[1]] = tensors[hkey]
        hidden[old_count:, :new[hkey].shape[1]] = new[hkey]
        mask[:old_count, :tensors[mkey].shape[1]] = tensors[mkey]
        mask[old_count:, :new[mkey].shape[1]] = new[mkey]
        tensors[hkey], tensors[mkey] = hidden, mask
        metadata["token_ids"][prefix] += details["token_ids"][prefix]
        metadata["lengths"][prefix] += details["lengths"][prefix]
    metadata["spec"] = spec
    return tensors, metadata


class ReactiveGWMPipeline:
    """Load once, then generate from a CSV sample or image/masks/controls."""

    @classmethod
    def from_pretrained(cls, checkpoint, *, subfolder=None, revision=None, cache_dir=None,
                        config=None, device="cuda", precision=None, vae=None,
                        text_encoder=None, tokenizer=None):
        from .state import load_weights

        directory, source = resolve_checkpoint(checkpoint, subfolder=subfolder, revision=revision, cache_dir=cache_dir)
        settings = load_config(config or directory / "config.yaml")
        if precision:
            settings["training"]["precision"] = precision
        result = cls()
        result.config, result.source, result.checkpoint = settings, source, directory
        result.device, result.vae = device, vae
        result.text_encoder, result.tokenizer = text_encoder, tokenizer
        result.geometry = VideoGeometry(**settings["video"])
        result.model = make_model(settings, device).eval()
        validate_checkpoint(directory, settings, result.model)
        load_weights(result.model, directory)
        result.text_path = directory / "text/control_text.safetensors"
        return result

    def _vae_pipeline(self):
        if not self.vae:
            raise ValueError("provide --vae for raw first-frame encoding or video decoding")
        from .video import _load_pipeline

        return _load_pipeline([str(self.vae)], None, str(self.device))

    def sample_csv(self, data, *, split="val", index=0, sample_id=None, output,
                   text_cache=None, controls=None, seed=20260808, steps=None,
                   sigma_shift=None, latent_only=False):
        import torch
        from safetensors.torch import save_file
        from diffsynth.diffusion import FlowMatchScheduler
        from .csv_data import CSVIndex
        from .data import CachedDataset
        from .infer import condition_batch, sample_latents
        from .video import _encode_first_frame, write_video

        destination = Path(output).expanduser().resolve()
        if destination.exists():
            raise FileExistsError(f"choose a new sample output directory: {destination}")
        # Checkpoint text rows are authoritative; CSV controls map by actual text.
        selected_text = text_cache or self.text_path
        dataset_index = CSVIndex(data, text_cache=selected_text, control_steps=self.geometry.control_steps)
        rows = [r for r in dataset_index.records if r["split"] == split]
        if sample_id is not None:
            matches = [i for i, row in enumerate(rows) if row["record_id"] == sample_id]
            if len(matches) != 1:
                raise ValueError(f"sample ID not found uniquely in {split}: {sample_id}")
            index = matches[0]
        if not 0 <= index < len(rows):
            raise ValueError("sample index outside the selected split")
        if self.config["game"] == "sf" and len(rows[index]["subjects"]) != 2:
            raise ValueError("the released SF3 model requires exactly two fighters")
        if controls is not None:
            return self(
                image=rows[index]["first_frame"], masks=rows[index]["x0_masks"],
                controls=controls, output=output, text_cache=selected_text,
                seed=seed, steps=steps, sigma_shift=sigma_shift, latent_only=latent_only,
            )
        if not latent_only and not self.vae:
            raise ValueError("provide --vae or choose --latent-only")
        tensors, _ = extend_text(
            selected_text, dataset_index.text_spec(), encoder_path=self.text_encoder,
            tokenizer_path=self.tokenizer, device=self.device,
        )
        bind_text(self.model, tensors)
        del tensors
        dataset = CachedDataset(dataset_index, split, geometry=self.geometry, allow_uncached=True)
        row = dataset.records[index]
        first = None
        vae_pipe = None
        if not row["latents"]:
            if not row["first_frame"]:
                raise ValueError("uncached inference requires first_frame; future video is never read")
            vae_pipe = self._vae_pipeline()
            with torch.inference_mode():
                first = _encode_first_frame(vae_pipe, Path(row["first_frame"]), self.geometry)[0]
        item = dataset.inference_item(index, first_frame_latents=first)
        first, conditions = condition_batch(item, self.device, next(self.model.parameters()).dtype, method="pretraining")
        steps = steps if steps is not None else self.config["inference"]["denoising_steps"]
        shift = sigma_shift if sigma_shift is not None else self.config["inference"]["sigma_shift"]
        if steps < 1 or shift <= 0:
            raise ValueError("steps and sigma shift must be positive")
        scheduler = FlowMatchScheduler("Wan")
        scheduler.set_timesteps(steps, shift=shift)
        latent = sample_latents(
            self.model, first, conditions, scheduler, seed,
            progress=lambda n: print(f"Denoising {n}/{steps}", flush=True),
        )
        destination.mkdir(parents=True)
        save_file({"latents": latent[0].cpu().contiguous()}, str(destination / "latents.safetensors"))
        write_json(destination / "settings.json", {
            "sample_id": row["record_id"], "game": self.config["game"], "checkpoint": self.source,
            "video": self.config["video"], "seed": seed, "denoising_steps": steps,
            "sigma_shift": shift, "cfg_scale": 1,
            "subjects": row["subjects"], "control_kind": row["control_kind"],
            "action_ids": row["action_ids"], "npc_prompts": row["npc_prompts"],
            "text_cache": str(Path(selected_text).resolve()), "action_vocabulary": dataset_index.actions,
        })
        if not latent_only:
            vae_pipe = vae_pipe or self._vae_pipeline()
            with torch.inference_mode():
                decoded = vae_pipe.vae.decode(
                    latent.to(dtype=torch.bfloat16), device=self.device, tiled=True,
                    tile_size=self.geometry.latent_size,
                    tile_stride=tuple(x // 2 for x in self.geometry.latent_size),
                )
            if not torch.isfinite(decoded).all():
                raise FloatingPointError("nonfinite decoded video")
            write_video(destination / "video.mp4", vae_pipe.vae_output_to_video(decoded), geometry=self.geometry)
        print(f"Generated {destination}", flush=True)
        return destination

    def __call__(self, *, image, masks, controls, output, **kwargs):
        """Each controls.subjects entry supplies actions OR a prompt, in mask order."""
        if isinstance(controls, (str, Path)):
            controls = json.loads(Path(controls).read_text(encoding="utf-8"))
        if not image:
            raise ValueError("provide a first-frame image when editing sample controls")
        with tempfile.TemporaryDirectory(prefix="reactivegwm-input-") as temporary:
            root = Path(temporary)
            write_json(root / "controls.json", controls)
            with (root / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
                row = {
                    "sample_id": "custom", "split": "test", "video": "", "latents": "",
                    "first_frame": str(Path(image).expanduser().resolve()),
                    "masks": json.dumps([str(Path(p).expanduser().resolve()) for p in masks]),
                    "controls": "controls.json", "text_cache": "",
                }
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            return self.sample_csv(root / "samples.csv", split="test", output=output, **kwargs)
