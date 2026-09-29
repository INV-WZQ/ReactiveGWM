"""Release interfaces: effective batch, HF archives, raw inputs and new text."""
import csv
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from safetensors.torch import save_file

from fixtures import recipe, create_cache, batch
from selective_agency.runtime.config import resolve_batch_size, make_model
from selective_agency.runtime.hf_data import archive_plan, extract_selected, member_path
from selective_agency.runtime.pipeline import ReactiveGWMPipeline, extend_text
from selective_agency.runtime.text import load_text_asset


class ReleaseInterfaces(unittest.TestCase):
    def test_batch_defaults_and_overrides(self):
        for world, accumulation in [(1, 8), (2, 4), (4, 2), (8, 1)]:
            settings = recipe()["training"]
            settings["gradient_accumulation_steps"] = None
            resolve_batch_size(settings, world)
            self.assertEqual(settings["gradient_accumulation_steps"], accumulation)
            self.assertEqual(settings["effective_batch_size"], 8)
        settings = recipe()["training"]
        settings.update(batch_size=2, gradient_accumulation_steps=3)
        resolve_batch_size(settings, 2)
        self.assertEqual(settings["effective_batch_size"], 12)
        settings.update(gradient_accumulation_steps=None, effective_batch_size=7)
        with self.assertRaises(ValueError):
            resolve_batch_size(settings, 2)

    def test_extracts_only_named_regular_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "samples.tar"
            with tarfile.open(archive, "w") as handle:
                for name, value in [("HNM/x0.png", b"first"), ("unselected", b"unused")]:
                    info = tarfile.TarInfo(name)
                    info.size = len(value)
                    handle.addfile(info, io.BytesIO(value))
                link = tarfile.TarInfo("HNM/link")
                link.type, link.linkname = tarfile.SYMTYPE, "/etc/passwd"
                handle.addfile(link)
            extract_selected(archive, root / "out", ["HNM/x0.png"])
            self.assertEqual((root / "out/HNM/x0.png").read_bytes(), b"first")
            self.assertFalse((root / "out/unselected").exists())
            with self.assertRaises(ValueError):
                extract_selected(archive, root / "out", ["HNM/link"])
            with self.assertRaises(FileNotFoundError):
                extract_selected(archive, root / "out", ["absent"])
            for name in ["../escape", "/absolute", "C:/absolute", "a\\b"]:
                with self.assertRaises(ValueError):
                    member_path(root, name)

    def test_test_split_fetches_annotations_and_shared_text_without_video(self):
        row = dict(archive="HNM/test/a.tar", masks_members_json='["HNM/mask.png"]',
                   first_frame_member="HNM/first.png", controls_member="", latents_member="",
                   annotation_archive="HNM/shared/part-00001.tar",
                   annotation_member="HNM/dataset/test/metadata.jsonl", video_member="HNM/video.mkv")
        plan = archive_plan([row], "HNM")
        self.assertNotIn("HNM/video.mkv", plan[row["archive"]])
        self.assertIn(row["annotation_member"], plan[row["annotation_archive"]])
        self.assertIn("HNM/cache/pretraining/control_text.json", plan[row["annotation_archive"]])

    def test_raw_inputs_do_not_need_future_video_and_keep_anchor(self):
        from PIL import Image
        from safetensors.torch import load_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_cache(root / "cache", "hnm")
            config = recipe()
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "config.yaml").write_text(yaml.safe_dump(config))
            save_file(make_model(config, "cpu").state_dict(), str(checkpoint / "model.safetensors"))
            Image.new("RGB", (64, 64)).save(root / "first.png")
            text = root / "cache/control_text.safetensors"
            controls = {"subjects": [
                {"name": "P1", "actions": ["jump"] * 25},
                {"name": "P2", "prompt": "If the opponent approaches, then block."},
            ]}
            first = batch()["first_frame_latents"]
            pipe = ReactiveGWMPipeline.from_pretrained(checkpoint, device="cpu")
            with patch.object(pipe, "_vae_pipeline", return_value=object()), patch(
                "selective_agency.runtime.video._encode_first_frame", return_value=first
            ) as encode:
                pipe(image=root / "first.png", masks=[root / "cache/mask-0.png", root / "cache/mask-1.png"],
                     controls=controls, output=root / "result", text_cache=text, steps=2, latent_only=True)
            encode.assert_called_once()
            latent = load_file(str(root / "result/latents.safetensors"))["latents"]
            torch.testing.assert_close(latent[:, :1], first[0], atol=0, rtol=0)
            settings = json.loads((root / "result/settings.json").read_text())
            self.assertEqual(settings["control_kind"], [1, 2])
            # Swapping roles is expressed by controls order, with masks left bound to physical subjects.
            controls["subjects"] = [
                {"name": "P1", "prompt": "If the opponent approaches, then block."},
                {"name": "P2", "actions": ["jump"] * 25},
            ]
            with patch.object(pipe, "_vae_pipeline", return_value=object()), patch(
                "selective_agency.runtime.video._encode_first_frame", return_value=first
            ):
                pipe(image=root / "first.png", masks=[root / "cache/mask-0.png", root / "cache/mask-1.png"],
                     controls=controls, output=root / "swapped", text_cache=text, steps=2, latent_only=True)
            changed = json.loads((root / "swapped/settings.json").read_text())
            self.assertEqual(changed["control_kind"], [2, 1])

    def test_new_text_requires_encoder_and_preserves_existing_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_cache(root, "hnm")
            path = root / "control_text.safetensors"
            old, metadata = load_text_asset(path)
            spec = {**metadata["spec"], "npc_prompts": metadata["spec"]["npc_prompts"] + ["A new conditional rule."]}
            with self.assertRaisesRegex(ValueError, "new text"):
                extend_text(path, spec, encoder_path=None, tokenizer_path=None, device="cpu")
            def encode(spec, **kwargs):
                tensors = {
                    "action_hidden_states": torch.ones(1, 2, 16, dtype=torch.bfloat16),
                    "action_attention_mask": torch.ones(1, 2, dtype=torch.bool),
                    "npc_hidden_states": torch.ones(1, 10, 16, dtype=torch.bfloat16),
                    "npc_attention_mask": torch.ones(1, 10, dtype=torch.bool),
                }
                return tensors, {"token_ids": {"action": [[1, 2]], "npc": [list(range(10))]},
                                 "lengths": {"action": [2], "npc": [10]}}
            with patch("selective_agency.runtime.text.load_frozen_wan_text_encoder", return_value=object()), patch(
                "selective_agency.runtime.text.load_full_text_tokenizer", return_value=object()
            ), patch("selective_agency.runtime.text.encode_text_spec", side_effect=encode):
                new, details = extend_text(path, spec, encoder_path="encoder", tokenizer_path="tokenizer", device="cpu")
            torch.testing.assert_close(old["action_hidden_states"], new["action_hidden_states"], atol=0, rtol=0)
            torch.testing.assert_close(old["npc_hidden_states"], new["npc_hidden_states"][:2, :7], atol=0, rtol=0)
            self.assertEqual(details["lengths"]["npc"], [7, 7, 10])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
