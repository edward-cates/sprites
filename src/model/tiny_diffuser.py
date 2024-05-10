import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm
import torch

from src.model.noise_scheduler import GaussianDiffusion

class TinyDiffuser(torch.nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.noise_scheduler = GaussianDiffusion(timesteps=1000)
        self.conv_in = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, 64, kernel_size=5, stride=1, padding=2, bias=False),
            torch.nn.ReLU(),
        )
        self.conv_mid = torch.nn.Sequential(
            torch.nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.ReLU(),
        )
        self.conv_out = torch.nn.Conv3d(64, in_channels, kernel_size=3, stride=1, padding=1, bias=False)

    @property
    def total_timesteps(self) -> int:
        return self.noise_scheduler.num_timesteps

    def forward(self, x):
        # input: signal,
        # output: predicted noise.
        # x, multichannel.
        latent = self.conv_in(x)
        # consider this a residual.
        d = self.conv_mid(latent)
        # skip connection
        latent = latent + d
        # find the residual again, but this time, return the residual as the noise.
        d = self.conv_out(latent)
        assert x.shape == d.shape, f"Expected {x.shape} and {d.shape} to match."
        return d

    def create_noised_image(self, x: torch.Tensor, t: Optional[int] = None) -> torch.Tensor:
        b = x.shape[0]
        if t is not None:
            # expand to shape (b,)
            t = torch.full((b,), t, device=x.device, dtype=torch.long)
        else:
            # random integers from 0 to total_timesteps, size (b,)
            t = self.noise_scheduler._sample_random_times(b, device=x.device)

        noise = torch.randn_like(x)
        return (
            self.noise_scheduler.q_sample(x, t),
            t,
            noise,
        )

    def generate(self, example_x: torch.Tensor) -> torch.Tensor:
        b = example_x.shape[0]
        sampling_timesteps = self.noise_scheduler._get_sampling_timesteps(b, device=example_x.device)
        img = torch.randn_like(example_x)
        with torch.no_grad():
            for t in tqdm(sampling_timesteps, desc="Generating video"):
                predicted_noise = self(img)
                # x_0 = self.noise_scheduler.predict_start_from_noise(
                #     x_t = img,
                #     t = t,
                #     noise = predicted_noise,
                # )
                x_0 = img - predicted_noise
                x_t_1 = self.noise_scheduler.q_posterior(
                    x_start = x_0,
                    x_t = img,
                    t = t,
                )
                img = x_t_1
        return img


if __name__ == "__main__":
    model = TinyDiffuser()
    print(model)

    device = "cuda:1"
    model.to(device)

    x = torch.randn(1, 3, 8, 64, 64).to(device)
    p_noise = model(x)

    assert x.shape == p_noise.shape, f"Expected {x.shape} and {p_noise.shape} to match."
    print("Success!")
