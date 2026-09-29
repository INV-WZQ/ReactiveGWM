"""Only-if-then inference; the only video condition is the clean first frame."""

from contextlib import nullcontext


def condition_batch(item, device, dtype, *, method):
    from selective_agency.data.collate import collate_samples
    from .batch import move_batch

    # Collation also checks physical roles and masks. This placeholder contains
    # only the first frame and is discarded before the model is called.
    frames = item["action_ids"].shape[0] + 1
    value = dict(item, input_latents=item["first_frame_latents"].expand(-1, frames, -1, -1))
    batch = move_batch(collate_samples([value]), device, dtype)
    conditions = {
        name: batch[name]
        for name in (
            "subject_roi_masks",
            "action_ids",
            "subject_valid",
            "control_kind",
            "npc_prompt_ids",
        )
    }
    return batch["first_frame_latents"], conditions


def sample_latents(model, first, conditions, scheduler, seed, progress=None):
    """Original Wan schedule, noise precision and anchor replacement at every step."""
    import torch

    device = first.device
    generator = torch.Generator(device=device).manual_seed(seed)
    shape = (first.shape[0], first.shape[1], conditions["action_ids"].shape[1] + 1, *first.shape[-2:])
    with torch.inference_mode():
        latent = torch.randn(shape, generator=generator, device=device, dtype=first.dtype)
        latent[:, :, :1] = first
        for number, timestep in enumerate(scheduler.timesteps, 1):
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda" and first.dtype == torch.bfloat16
                else nullcontext()
            )
            with context:
                prediction = model(
                    latent,
                    timestep.to(device=device, dtype=first.dtype).expand(first.shape[0]),
                    **conditions,
                    return_routing_aux=False,
                )
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("nonfinite inference prediction")
            latent = scheduler.step(prediction, timestep, latent)
            latent[:, :, :1] = first
            if progress:
                progress(number)
        if not torch.isfinite(latent).all() or not torch.equal(latent[:, :, :1], first):
            raise FloatingPointError("inference failed to preserve the clean first-frame anchor")
    return latent
