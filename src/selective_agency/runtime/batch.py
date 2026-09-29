"""Move input tensors without importing the training loop."""

def move_batch(batch, device, dtype):
    import torch

    return {
        key: (
            value.to(
                device=device,
                dtype=dtype if key in {"input_latents", "first_frame_latents"} else value.dtype,
            )
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }
