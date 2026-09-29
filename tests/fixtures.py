"""Small real-model inputs used by the numerical and resume checks."""

from pathlib import Path
import sys
import torch

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING / "src"))

SMALL_MODEL = dict(
    dim=32, in_dim=4, out_dim=4, ffn_dim=64, text_dim=16, freq_dim=8, num_heads=4, num_layers=2
)


def recipe(game="hnm", method="pretraining", precision="fp32", *, control_steps=25):
    from selective_agency.runtime.config import load_config

    value = load_config(TRAINING / "training/configs" / ("sf3.yaml" if game == "sf" else "hnm.yaml"))
    value["method"] = method
    value["model"].update(SMALL_MODEL)
    value["model"]["control_steps"] = control_steps
    value["video"].update(width=64, height=64, frames=4 * control_steps + 1)
    value["training"].update(precision=precision, num_workers=0, gradient_accumulation_steps=1)
    return value


def text_tensors():
    generator = torch.Generator().manual_seed(715)
    tensors = {
        "action_hidden_states": torch.randn(3, 4, 16, generator=generator).to(torch.bfloat16),
        "action_attention_mask": torch.ones(3, 4, dtype=torch.bool),
        "npc_hidden_states": torch.randn(2, 7, 16, generator=generator).to(torch.bfloat16),
        "npc_attention_mask": torch.ones(2, 7, dtype=torch.bool),
    }
    return tensors


def batch(dtype=torch.float32, *, control_steps=25):
    generator = torch.Generator().manual_seed(611)
    actions = torch.full((1, control_steps, 6), -1, dtype=torch.long)
    actions[0, :, 0] = torch.arange(control_steps) % 3
    masks = torch.zeros(1, 6, 2, 2)
    masks[0, 0, 0, 0] = 1
    masks[0, 1, 1, 1] = 1
    return {
        "input_latents": torch.randn(1, 4, control_steps + 1, 4, 4, generator=generator).to(dtype),
        "first_frame_latents": torch.randn(1, 4, 1, 4, 4, generator=generator).to(dtype),
        "subject_roi_masks": masks,
        "action_ids": actions,
        "subject_valid": torch.tensor([[True, True, False, False, False, False]]),
        "control_kind": torch.tensor([[1, 2, 0, 0, 0, 0]]),
        "npc_prompt_ids": torch.tensor([[-1, 0, -1, -1, -1, -1]]),
    }


def forward(model, value):
    return model(
        value["input_latents"],
        torch.tensor(
            [500], dtype=value["input_latents"].dtype, device=value["input_latents"].device
        ),
        value["subject_roi_masks"],
        value["action_ids"],
        value["subject_valid"],
        control_kind=value["control_kind"],
        npc_prompt_ids=value["npc_prompt_ids"],
    )


def create_cache(root, game, *, control_steps=25, dtype=torch.bfloat16, include_validation=True):
    import numpy as np
    from PIL import Image
    from safetensors.torch import save_file
    from selective_agency.runtime.io import write_json
    from selective_agency.runtime.csv_data import write_dataset_csv
    from selective_agency.runtime.text import save_text_asset

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tensors = text_tensors()
    protocol = "hnm-if-then" if game == "hnm" else "sf-if-then"
    prompts = ["If the opponent approaches, then block.", "If the opponent retreats, then follow."]
    actions = ["no_op", "jump", "punch"]
    spec = dict(
        protocol="selective-agency-full-condition-text",
        prompt_protocol=protocol,
        actions=actions,
        action_texts=[f"ACTION={x}" for x in actions],
        npc_prompts=prompts,
        action_protocol="tiny-test-actions",
        raw_id_to_row={str(i): i for i in range(3)},
    )
    text = dict(
        spec=spec,
        protocol=spec["protocol"],
        text_dim=16,
        token_ids={"action": [[1, 2, 3, 4]] * 3, "npc": [[1, 2, 3, 4, 5, 6, 7]] * 2},
        lengths={"action": [4] * 3, "npc": [7] * 2},
        tokenization={"truncation": False},
    )
    save_text_asset(root, tensors, text)
    for slot in range(2):
        pixels = np.zeros((8, 8), dtype=np.uint8)
        pixels[0:4, 0:4] = 255 if slot == 0 else 0
        pixels[4:8, 4:8] = 255 if slot == 1 else 0
        Image.fromarray(pixels).save(root / f"mask-{slot}.png")
    records = []
    splits = (("train", 5), ("val", 2)) if include_validation else (("train", 5),)
    for split, count in splits:
        for index in range(count):
            value = batch(dtype, control_steps=control_steps)
            video = value["input_latents"][0] + index / 10
            save_file(
                {"input_latents": video, "first_frame_latents": value["first_frame_latents"][0]},
                str(root / f"{split}-{index}.safetensors"),
            )
            records.append(
                dict(
                    record_id=f"test:{split}:{index}",
                    split=split,
                    sample_index=index,
                    family_id=f"{split}-{index}",
                    branch="main",
                    subjects=["P1", "P2"],
                    control_kind=[1, 2],
                    action_ids=value["action_ids"][0, :, :2].tolist(),
                    npc_prompts=[None, prompts[index % 2]],
                    source_identity=["test", split, str(index)],
                    source_group_identity=["test", f"{split}-{index}"],
                    x0_masks=[str(root / "mask-0.png"), str(root / "mask-1.png")],
                    latents=str(root / f"{split}-{index}.safetensors"),
                )
            )
    write_dataset_csv(root / "samples.csv", records, actions, text_cache=root / "control_text.safetensors")
