"""Run with TRAINING_GAME=hnm or TRAINING_GAME=sf in separate processes."""

import os
from pathlib import Path
import tempfile
import unittest
import torch

from fixtures import recipe, text_tensors, batch, forward, create_cache
from selective_agency.runtime.config import select_core, make_model, bind_text

GAME = os.environ.get("TRAINING_GAME", "hnm")
select_core(GAME)
from selective_agency.loss import WanFlowMatchingObjective
from selective_agency.sampler import StatefulDistributedSampler
from selective_agency.runtime.data import CachedDataset
from selective_agency.runtime.infer import condition_batch, sample_latents

torch.set_num_threads(1)


def make(method="pretraining"):
    torch.manual_seed(17)
    model = make_model(recipe(GAME, method), "cpu", trainable=True)
    bind_text(model, text_tensors())
    return model


class CoreBehavior(unittest.TestCase):
    def test_games_coexist(self):
        models = []
        for game, method in (("hnm", "pretraining"), ("sf", "pretraining")):
            select_core(game)
            torch.manual_seed(17)
            model = make_model(recipe(game, method), "cpu").eval()
            bind_text(model, text_tensors())
            models.append(model)
        value = batch()
        with torch.no_grad():
            hnm, sf = [forward(model, value) for model in models]
        torch.testing.assert_close(hnm, sf, rtol=0, atol=0)
        self.assertEqual(models[0].architecture_version, "native_causal_handle_full_condition_fm")
        self.assertEqual(models[1].architecture_version, "native_causal_handle_actor_prompt_fm")

    def test_causal_model_excludes_future_frames(self):
        model = make().eval()
        original = batch()
        changed = {**original, "input_latents": original["input_latents"].clone()}
        changed["input_latents"][:, :, -1] += 3
        with torch.no_grad():
            a, b = forward(model, original), forward(model, changed)
        torch.testing.assert_close(a[:, :, :-1], b[:, :, :-1], rtol=0, atol=0)
        self.assertFalse(torch.equal(a[:, :, -1], b[:, :, -1]))




    def test_flow_loss_ignores_anchor_prediction(self):
        value = batch()
        noise = torch.ones_like(value["input_latents"])
        target = noise - value["input_latents"]
        target[:, :, 0] = 12345

        class ExactFuture(torch.nn.Module):
            def forward(self, latent, *args, **kwargs):
                torch.testing.assert_close(
                    latent[:, :, :1], value["first_frame_latents"], rtol=0, atol=0
                )
                return target

        result = WanFlowMatchingObjective()(
            ExactFuture(), value, use_gradient_checkpointing=False, timestep_id=123, noise=noise
        )
        self.assertEqual(result.loss.item(), 0)

    def test_sampler_prefetch_does_not_commit_progress(self):
        sampler = StatefulDistributedSampler(11, num_replicas=2, rank=0, seed=18)
        iterator = iter(sampler)
        first, second = next(iterator), next(iterator)
        self.assertEqual(sampler.cursor, 0)
        sampler.mark_consumed(1)
        resumed = StatefulDistributedSampler(11, num_replicas=2, rank=0, seed=18)
        resumed.load_state_dict(sampler.state_dict())
        self.assertEqual(next(iter(resumed)), second)
        self.assertNotEqual(first, second)

    def test_inference_reads_only_first_frame_and_preserves_it(self):
        from safetensors.torch import save_file
        from diffsynth.diffusion import FlowMatchScheduler

        with tempfile.TemporaryDirectory(prefix="training-core-") as temporary:
            root = Path(temporary)
            create_cache(root, GAME)
            dataset = CachedDataset(root, "val")
            item = dataset.inference_item(0)
            # Remove the future-video tensor completely: inference must still run.
            save_file(
                {"first_frame_latents": item["first_frame_latents"]},
                str(root / "val-0.safetensors"),
            )
            reread = dataset.inference_item(0)
            self.assertNotIn("input_latents", reread)
            first, conditions = condition_batch(reread, "cpu", torch.float32, method="pretraining")
            model = make().eval()
            scheduler = FlowMatchScheduler("Wan")
            scheduler.set_timesteps(2, shift=5)
            result = sample_latents(model, first, conditions, scheduler, 13)
            torch.testing.assert_close(result[:, :, :1], first, rtol=0, atol=0)

    def test_reshard_reseeds_reproducibly_at_large_saved_steps(self):
        import numpy as np
        from accelerate import Accelerator
        from selective_agency.runtime.io import write_json
        from selective_agency.runtime.state import TrainingProgress, restore

        config = recipe(GAME)
        model = make()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1 / 3, total_iters=5
        )
        accelerator = Accelerator(cpu=True)
        old_sampler = StatefulDistributedSampler(11, num_replicas=2, rank=0, seed=20260808)
        old_sampler.mark_consumed(1)
        with tempfile.TemporaryDirectory(prefix="training-large-step-") as temporary:
            root = Path(temporary)
            torch.save(model.state_dict(), root / "pytorch_model.bin")
            torch.save(optimizer.state_dict(), root / "optimizer.bin")
            torch.save(scheduler.state_dict(), root / "scheduler.bin")
            write_json(root / "sampler-rank-0000.json", old_sampler.state_dict())
            for step in (28000, 40000):
                with self.subTest(step=step):
                    torch.save(
                        {"global_step": step, "samples_seen": step * 8},
                        root / "custom_checkpoint_0.pkl",
                    )
                    draws = []
                    for prior_seed in (17, 29):
                        np.random.seed(prior_seed)
                        torch.manual_seed(prior_seed)
                        progress = TrainingProgress()
                        sampler = StatefulDistributedSampler(
                            11, num_replicas=1, rank=0, seed=20260808
                        )
                        restore(
                            root,
                            "reshard",
                            accelerator=accelerator,
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            progress=progress,
                            sampler=sampler,
                            config=config,
                        )
                        self.assertEqual(progress.step, step)
                        self.assertEqual(progress.samples_seen, step * 8)
                        self.assertEqual(sampler.cursor, 2)
                        draws.append((np.random.random(), torch.rand(3)))
                    self.assertEqual(draws[0][0], draws[1][0])
                    torch.testing.assert_close(draws[0][1], draws[1][1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
