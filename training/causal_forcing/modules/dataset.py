"""CF dataset builder — SF3 21-latent-frame 训练窗口.

直接复用 `diffsynth.core.UnifiedDataset` + `ReactiveGWM_Code.training.data` 操作算子,
schema 与普通双向 SFT 训练完全一致 (CSV metadata + video parquet action).

输出每个 row: `{video: [PIL.Image, ...], action: [1, T, num_buttons], prompt: str?}`
后续 CFARTrainingModule.forward 用 `pipe.unit_runner` 把 video/prompt 跑 VAE/T5 encode.
"""

from __future__ import annotations

from typing import Sequence

from diffsynth.core import UnifiedDataset


def build_cf_dataset(
    *,
    metadata_path: str,
    dataset_base: str,
    game: str = "sf3",
    num_frames: int = 101,
    height: int = 480,
    width: int = 832,
    action_hold_window: int = 10,
    dataset_repeat: int = 100,
    data_file_keys: Sequence[str] = ("video", "action"),
    max_pixels: int = 1920 * 1080,
) -> UnifiedDataset:
    """Build CF Stage 1 / 2 / 3 共用 SF3 dataset.

    复用普通双向训练已经迁移的数据 profile / action operator。
    """
    # Lazy import keeps the dataset helper cheap to import in --help paths.
    from ReactiveGWM_Code.training.data.action_utils import get_action_op  # noqa: WPS433
    from ReactiveGWM_Code.training.data.profiles import get_profile  # noqa: WPS433

    profile = get_profile(game)
    return UnifiedDataset(
        base_path=dataset_base,
        metadata_path=metadata_path,
        repeat=dataset_repeat,
        data_file_keys=list(data_file_keys),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=dataset_base,
            max_pixels=max_pixels,
            height=height,
            width=width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "action": get_action_op(
                profile, dataset_base, num_frames, hold_window=action_hold_window
            ),
        },
    )
