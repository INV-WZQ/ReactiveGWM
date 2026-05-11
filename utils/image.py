"""Image / video tensor conversions and mp4 IO."""
from typing import List

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import InterpolationMode


def round_up(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def _crop_and_resize_like_training(img: Image.Image, target_height: int,
                                    target_width: int) -> Image.Image:
    """Aspect-preserving fit-and-center-crop, bit-identical to the training
    dataset operator (`diffsynth/core/data/operators.py:ImageCropAndResize.
    crop_and_resize`): scale so both dims are ≥ target via torchvision BILINEAR,
    then center-crop to (target_height, target_width). On matched-aspect inputs
    (e.g. SF clips at native resolution) this degenerates to identity.
    """
    src_w, src_h = img.size                                # PIL: (W, H)
    scale = max(target_width / src_w, target_height / src_h)
    img = TF.resize(
        img, [round(src_h * scale), round(src_w * scale)],
        interpolation=InterpolationMode.BILINEAR,
    )
    img = TF.center_crop(img, [target_height, target_width])
    return img


def preprocess_image(img: Image.Image, height: int, width: int,
                     device: str, dtype: torch.dtype) -> torch.Tensor:
    """PIL → torch [3, 1, H, W] in [-1, 1].

    Aspect-preserving resize + center crop, matching the training dataset's
    `ImageCropAndResize` operator (resize BILINEAR → center_crop). For
    matched-aspect input (SF clips at native resolution) this is identity.
    For mismatched aspect the input is scaled to cover (target_h, target_w)
    and excess is cropped — same as training time.

    The cast-then-scale order is also intentional: it matches what
    `base_pipeline.preprocess_image` does in DiffSynth-Studio, which is what
    `precompute_cache.py` (cached training) and `WanVideoUnit_ImageEmbedderFused`
    (non-cached training) both use. Keeping the scale in `dtype` (bf16) makes the
    first-frame latent bit-identical to the one the model saw at training time.
    Don't "optimize" this back to fp32 — see analysis 3.2/④.
    """
    img = img.convert("RGB")
    img = _crop_and_resize_like_training(img, height, width)
    arr = torch.from_numpy(np.array(img, dtype=np.float32))
    arr = arr.permute(2, 0, 1).unsqueeze(1)              # [3, 1, H, W], fp32 CPU
    arr = arr.to(device=device, dtype=dtype)             # → bf16 GPU (cast first)
    arr = arr * (2.0 / 255.0) - 1.0                      # scale in bf16
    return arr


def to_pil_video(video: torch.Tensor) -> List[Image.Image]:
    """[1, 3, T, H, W] in [-1, 1] → list[PIL.Image]."""
    v = video[0].clamp(-1, 1)
    v = ((v + 1) * 127.5).round().to(torch.uint8)
    v = v.permute(1, 2, 3, 0).contiguous().cpu().numpy()
    return [Image.fromarray(f) for f in v]


def save_mp4(frames: List[Image.Image], path: str, fps: int = 20, quality: int = 8):
    with imageio.get_writer(str(path), fps=fps, quality=quality) as w:
        for f in frames:
            w.append_data(np.array(f))
