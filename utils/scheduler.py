"""Flow-matching Euler scheduler (Wan template). Single-step inference.

Mirrors training-time `FlowMatchScheduler(template="Wan")` so the noise
schedule and `step()` math are bit-identical to what the model was trained on.
"""
import torch


class FlowMatchScheduler:
    def __init__(self, num_train_timesteps: int = 1000):
        self.num_train_timesteps = num_train_timesteps
        self.sigmas = None
        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int = 30, shift: float = 5.0,
                      denoising_strength: float = 1.0):
        sigma_min, sigma_max = 0.0, 1.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * self.num_train_timesteps

    def step(self, model_output: torch.Tensor, timestep: torch.Tensor,
             sample: torch.Tensor) -> torch.Tensor:
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        idx = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[idx]
        sigma_next = self.sigmas[idx + 1] if idx + 1 < len(self.timesteps) else 0.0
        return sample + model_output * (sigma_next - sigma)
