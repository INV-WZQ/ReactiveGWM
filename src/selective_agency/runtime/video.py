"""Original Wan preprocessing and VAE encoding."""

from pathlib import Path
from fractions import Fraction
import torch
from PIL import Image

from .geometry import DEFAULT_VIDEO


def _load_pipeline(model_paths: list[str], tokenizer_path: str | None, device: str):
    # Kept for the existing evaluation imports; loading is lazy for CPU checks.
    from diffsynth.core import ModelConfig
    from diffsynth.pipelines.wan_video import WanVideoPipeline

    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(path=path) for path in model_paths],
        tokenizer_config=None if tokenizer_path is None else ModelConfig(path=tokenizer_path),
        redirect_common_files=False,
    )


def _read_video(record, geometry=DEFAULT_VIDEO) -> list[Image.Image]:
    import av
    import numpy as np

    frames: list[Image.Image] = []
    with av.open(str(record.rgb_video)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        if (stream.width, stream.height) != (geometry.width, geometry.height):
            raise RuntimeError(f"source video must match configured size {geometry.width}x{geometry.height}")
        frames = [frame.to_image() for frame in container.decode(video=0)]
    if len(frames) != geometry.frames:
        raise RuntimeError(
            f"source video must contain {geometry.frames} frames for {geometry.control_steps} action steps"
        )
    if record.x0_rgb is not None:
        with Image.open(record.x0_rgb) as first:
            if first.size != (geometry.width, geometry.height) or not np.array_equal(
                np.asarray(first.convert("RGB")), np.asarray(frames[0])
            ):
                raise RuntimeError("source first frame differs from decoded video frame 0; leave first_frame empty to extract it")
    return frames


def _encode_video(pipe, frames: list[Image.Image]) -> torch.Tensor:
    video = pipe.preprocess_video(frames)
    return pipe.vae.encode(video, device=pipe.device, tiled=False).to(
        dtype=torch.bfloat16, device="cpu"
    )


def _encode_first_frame(pipe, path: Path, geometry=DEFAULT_VIDEO) -> torch.Tensor:
    with Image.open(path) as image:
        if image.size != (geometry.width, geometry.height):
            raise RuntimeError("first frame geometry differs from video configuration")
        value = pipe.preprocess_image(image.convert("RGB")).transpose(0, 1)
    return pipe.vae.encode([value], device=pipe.device, tiled=False).to(
        dtype=torch.bfloat16, device="cpu"
    )


def write_video(path, frames, *, geometry=DEFAULT_VIDEO, fps=None):
    import av
    import numpy as np

    if len(frames) != geometry.frames or any(
        frame.size != (geometry.width, geometry.height) for frame in frames
    ):
        raise ValueError("decoded video frame count and size must match video configuration")
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(str(geometry.fps if fps is None else fps)))
        stream.width, stream.height, stream.pix_fmt = geometry.width, geometry.height, "yuv420p"
        stream.options = {"crf": "18"}
        for image in frames:
            frame = av.VideoFrame.from_ndarray(np.asarray(image.convert("RGB")), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
