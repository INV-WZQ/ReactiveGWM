"""CSV integration: portable locations, explicit controls and mixed encoding/reuse."""

import csv
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from safetensors.torch import save_file

from fixtures import create_cache, recipe
from selective_agency.runtime.config import select_core, load_config
from selective_agency.runtime.csv_data import CSVIndex
from selective_agency.runtime.data import CachedDataset
from selective_agency.runtime.cache import prepare
from selective_agency.runtime.cli import dataset_path
from selective_agency.runtime.geometry import VideoGeometry

GAME = os.environ.get("TRAINING_GAME", "hnm")
select_core(GAME)
from selective_agency.data.collate import collate_samples

torch.set_num_threads(1)


def edit_csv(path, change):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    change(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CSVBehavior(unittest.TestCase):
    def test_copy_and_reference_preserve_fp32_and_bf16_caches(self):
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input"
            create_cache(source, GAME)
            values = {key: value.float() for key, value in load_file(str(source / "train-0.safetensors")).items()}
            save_file(values, str(source / "train-0.safetensors"))
            config = recipe(GAME)
            geometry = VideoGeometry(**config["video"])
            original = CachedDataset(source, "train", geometry=geometry)
            for copy_assets in (True, False):
                output = root / ("copied" if copy_assets else "referenced")
                args = SimpleNamespace(
                    config=str(root / "recipe.yaml"), data=str(source), text_cache=None,
                    cache_root=str(output), split=["train"], limit=2,
                    encode_text=False, copy_assets=copy_assets, device="cpu",
                )
                prepare(args, config)
                prepared = CachedDataset(output, "train", geometry=geometry)
                for index in range(2):
                    for key in ("input_latents", "first_frame_latents"):
                        torch.testing.assert_close(prepared[index][key], original[index][key], rtol=0, atol=0)

    def test_video_config_rejects_misaligned_controls(self):
        import yaml

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "recipe.yaml"
            config = recipe(GAME)
            config["video"]["frames"] = 9
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ValueError, "control_steps"):
                load_config(path)
            del config["model"]["control_steps"]
            path.write_text(yaml.safe_dump(config))
            self.assertEqual(load_config(path)["model"]["control_steps"], 2)

    def test_raw_video_extracts_first_frame_with_custom_geometry(self):
        from selective_agency.runtime.video import write_video

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "input", root / "prepared"
            create_cache(source, GAME, control_steps=2)
            config = recipe(GAME, control_steps=2)
            geometry = VideoGeometry(**config["video"])
            first = Image.new("RGB", (geometry.width, geometry.height), (35, 117, 219))
            write_video(source / "clip.mp4", [first] * geometry.frames, geometry=geometry)
            edit_csv(source / "samples.csv", lambda rows: rows[0].update(
                video="clip.mp4", first_frame="", latents=""
            ))
            args = SimpleNamespace(
                config=str(root / "recipe.yaml"), data=str(source), text_cache=None,
                cache_root=str(output), split=["train"], limit=1, encode_text=False,
                copy_assets=True, device="cpu", vae="vae.pth",
            )
            video = torch.ones(1, 4, geometry.latent_frames, *geometry.latent_size)
            anchor = torch.zeros(1, 4, 1, *geometry.latent_size)
            with (
                patch("selective_agency.runtime.video._load_pipeline", return_value=object()),
                patch("selective_agency.runtime.video._encode_video", return_value=video) as encode_video,
                patch("selective_agency.runtime.video._encode_first_frame", return_value=anchor) as encode_first,
            ):
                prepare(args, config)
            frames = encode_video.call_args.args[1]
            self.assertEqual(len(frames), 9)
            dataset = CachedDataset(output, "train", geometry=geometry)
            extracted = Path(dataset.records[0]["first_frame"])
            self.assertTrue(extracted.is_relative_to(output))
            self.assertEqual(encode_first.call_args.args[1], extracted)
            with Image.open(extracted) as image:
                np.testing.assert_array_equal(np.asarray(image), np.asarray(frames[0]))
            item = dataset[0]
            self.assertEqual(item["action_ids"].shape, (2, 2))
            torch.testing.assert_close(item["input_latents"], video[0], rtol=0, atol=0)
            torch.testing.assert_close(item["first_frame_latents"], anchor[0], rtol=0, atol=0)

    def test_csv_path_from_config_and_command_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(config=str(root / "recipe.yaml"), data=None, cache_root=None)
            config = {"paths": {"data": "input.csv", "cache_root": "cache"}}
            self.assertEqual(dataset_path(args, config), root / "input.csv")
            for native_source in ("dataset", "indices/manifest.parquet"):
                config["paths"]["data"] = native_source
                self.assertEqual(dataset_path(args, config), root / "cache/samples.csv")
            config["paths"]["data"] = "input.csv"
            config["paths"]["dataset"] = "saved.csv"
            self.assertEqual(dataset_path(args, config), root / "saved.csv")
            args.cache_root = str(root / "override")
            self.assertEqual(dataset_path(args, config), root / "override/samples.csv")
            args.data = str(root / "new.csv")
            self.assertEqual(dataset_path(args, config), root / "new.csv")

    def test_new_ids_and_labels_need_no_dataset_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset with spaces"
            create_cache(root, GAME)
            metadata_path = root / "control_text.json"
            metadata = json.loads(metadata_path.read_text())
            metadata.pop("protocol", None)
            for key in ("protocol", "prompt_protocol", "action_protocol"):
                metadata["spec"].pop(key, None)
            metadata_path.write_text(json.dumps(metadata))
            def change(rows):
                for index, row in enumerate(rows):
                    row["sample_id"] = f'new collection, sample "{index}"'
                    row["dataset_version"] = "unregistered-source"
            edit_csv(root / "samples.csv", change)
            moved = root.with_name("relocated dataset")
            root.rename(moved)
            dataset = CachedDataset(moved / "samples.csv", "train")
            tensors, text = dataset.load_text(expected_dim=16)
            from selective_agency.runtime.text import save_text_asset, load_text_asset
            save_text_asset(moved / "saved_text", tensors, text)
            reloaded, _ = load_text_asset(moved / "saved_text", expected_dim=16)
            for key in tensors:
                torch.testing.assert_close(reloaded[key], tensors[key], rtol=0, atol=0)
            item = dataset.item_with_permutation(0, (1, 0))
            batch = collate_samples([item])
            self.assertEqual(item["record_id"], 'new collection, sample "0"')
            self.assertEqual(batch["control_kind"][0, :2].tolist(), [2, 1])
            self.assertEqual(batch["action_ids"][0, :, 1].tolist(), (torch.arange(25) % 3).tolist())
            self.assertEqual(batch["npc_prompt_ids"][0, :2].tolist(), [0, -1])

    def test_explicit_visibility_survives_role_permutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_cache(root, GAME)
            Image.fromarray(np.zeros((8, 8), dtype=np.uint8)).save(root / "empty.png")
            def change(rows):
                rows[0]["sample_id"] = "a-new-invisible-subject"
                masks = json.loads(rows[0]["masks"])
                masks[0] = "empty.png"
                rows[0]["masks"] = json.dumps(masks)
                control_path = root / rows[0]["controls"]
                controls = json.loads(control_path.read_text())
                controls["subjects"][0]["visible"] = False
                control_path.write_text(json.dumps(controls))
            edit_csv(root / "samples.csv", change)
            dataset = CachedDataset(root, "train")
            for permutation in ((0, 1), (1, 0)):
                batch = collate_samples([dataset.item_with_permutation(0, permutation)])
                slot = permutation.index(0)
                self.assertFalse(bool(batch["subject_roi_masks"][0, slot].any()))
                self.assertTrue(bool(batch["subject_valid"][0, slot]))

    def test_bad_action_length_reports_csv_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_cache(root, GAME)
            with (root / "samples.csv").open(newline="") as handle:
                first = next(csv.DictReader(handle))
            controls_path = root / first["controls"]
            controls = json.loads(controls_path.read_text())
            controls["subjects"][0]["actions"].pop()
            controls_path.write_text(json.dumps(controls))
            with self.assertRaisesRegex(ValueError, r"samples.csv:2:.*25"):
                CSVIndex(root)

    def test_mixed_raw_and_cached_rows_encode_only_missing_video(self):
        import av

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            create_cache(root, GAME)
            first = Image.fromarray(np.zeros((480, 832, 3), dtype=np.uint8))
            first.save(root / "first.png")
            with av.open(str(root / "video.mp4"), "w") as container:
                stream = container.add_stream("libx264", rate=20)
                stream.width, stream.height, stream.pix_fmt = 832, 480, "yuv420p"
                stream.options = {"crf": "0", "preset": "ultrafast"}
                for _ in range(101):
                    frame = av.VideoFrame.from_image(first)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            existing = {
                "input_latents": torch.full((48, 26, 30, 52), 3.0, dtype=torch.bfloat16),
                "first_frame_latents": torch.full((48, 1, 30, 52), 0.25, dtype=torch.bfloat16),
            }
            save_file(existing, str(root / "train-0.safetensors"))
            def change(rows):
                for row in rows:
                    row["text_cache"] = ""
                rows[1].update(video="video.mp4", first_frame="first.png", latents="")
            edit_csv(root / "samples.csv", change)
            output = Path(temporary) / "prepared"
            args = SimpleNamespace(
                config=str(root / "recipe.yaml"), data=str(root / "samples.csv"),
                cache_root=str(output), text_cache=None, split=["train"], limit=2,
                encode_text=False, copy_assets=True, tokenizer="tokenizer", text_encoder="encoder",
                device="cpu", text_batch_size=4, vae="vae.pth",
            )
            class Tokenizer:
                pad_token_id = 0
                model_max_length = 2
                def __call__(self, texts, **kwargs):
                    self.kwargs = kwargs
                    return {"input_ids": [[1] + [ord(char) % 31 + 2 for char in text] + [2] for text in texts]}
            class Encoder(torch.nn.Module):
                def forward(self, ids, mask):
                    return ids[..., None].expand(-1, -1, 16).float()
            tokenizer = Tokenizer()
            encoded_video = torch.full((1, 48, 26, 30, 52), 2.0, dtype=torch.bfloat16)
            encoded_first = torch.full((1, 48, 1, 30, 52), 0.5, dtype=torch.bfloat16)
            with (
                patch("selective_agency.runtime.text.load_full_text_tokenizer", return_value=tokenizer),
                patch("selective_agency.runtime.text.load_frozen_wan_text_encoder", return_value=Encoder()),
                patch("selective_agency.runtime.video._load_pipeline", return_value=object()) as loader,
                patch("selective_agency.runtime.video._encode_video", return_value=encoded_video) as video_encoder,
                patch("selective_agency.runtime.video._encode_first_frame", return_value=encoded_first) as first_encoder,
            ):
                config = recipe(GAME)
                config["video"].update(width=832, height=480)
                prepare(args, config)
            loader.assert_called_once()
            video_encoder.assert_called_once()
            first_encoder.assert_called_once()
            self.assertEqual(len(video_encoder.call_args.args[1]), 101)
            self.assertFalse(tokenizer.kwargs["truncation"])
            self.assertEqual(tokenizer.kwargs["padding"], False)
            dataset = CachedDataset(output, "train")
            dataset.load_text(expected_dim=16)
            torch.testing.assert_close(dataset[0]["input_latents"], existing["input_latents"], rtol=0, atol=0)
            torch.testing.assert_close(dataset[1]["input_latents"], encoded_video[0], rtol=0, atol=0)
            torch.testing.assert_close(dataset[1]["first_frame_latents"], encoded_first[0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
