"""Video dimensions and Wan's four-frame control intervals."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class VideoGeometry:
    width: int = 832
    height: int = 480
    frames: int = 101
    fps: float = 20

    def __post_init__(self):
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 32
            for value in (self.width, self.height)
        ):
            raise ValueError("video width and height must be positive multiples of 32")
        if (
            isinstance(self.frames, bool) or not isinstance(self.frames, int)
            or self.frames < 5 or (self.frames - 1) % 4
        ):
            raise ValueError("video.frames must be 4 * control_steps + 1, with at least one step")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("video.fps must be finite and positive")

    @property
    def control_steps(self):
        return (self.frames - 1) // 4

    @property
    def latent_frames(self):
        return self.control_steps + 1

    @property
    def latent_size(self):
        return self.height // 16, self.width // 16


DEFAULT_VIDEO = VideoGeometry()
