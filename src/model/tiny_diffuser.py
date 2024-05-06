import random

import numpy as np
import torch

class TinyDiffuser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.total_timesteps = 1000
        self.noise_schedule = [self.cosine_decay(t, self.total_timesteps) for t in range(self.total_timesteps)]
        self.layers = torch.nn.Sequential(
            torch.nn.Conv3d(3, 64, kernel_size=5, stride=1, padding=2, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.ReLU(),
            torch.nn.Conv3d(64, 3, kernel_size=3, stride=1, padding=1, bias=False),
        )

    @staticmethod
    def cosine_decay(t, total_timesteps):
        return 0.1 * (np.cos(np.pi * t / total_timesteps) + 1) / 2

    def create_noise(self, x: torch.Tensor) -> torch.Tensor:
        t = random.randint(0, self.total_timesteps - 1)
        noise_std = self.noise_schedule[t]
        noise = torch.randn_like(x) * noise_std
        return noise.to(x.device)

    def forward(self, x):
        # input: signal,
        # output: predicted noise.
        return self.layers(x)


if __name__ == "__main__":
    model = TinyDiffuser()
    print(model)

    device = "cuda:1"
    model.to(device)

    x = torch.randn(1, 3, 8, 64, 64).to(device)
    noise = model.create_noise(x)
    p_noise = model(x)

    assert x.shape == noise.shape == p_noise.shape, f"Expected {x.shape} and {p_noise.shape} to match."
    print("Success!")
