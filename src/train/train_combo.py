import random

from tqdm import tqdm
import wandb
import torch
from einops import rearrange

from src.model.tiny_diffuser import TinyDiffuser
from src.model.tiny_autoencoder import TinyAutoencoder

from src.dataset.flying_mnist_dataset import FlyingMnistDataset

class Trainer:
    def __init__(self, **kwargs):
        self.device = kwargs.get("device")

        self.vae = TinyAutoencoder()

        self.diffuser = TinyDiffuser()
        self.diffuser.to(device)

        self.train_dataset = FlyingMnistDataset("train")
        self.test_dataset = FlyingMnistDataset("val")

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=kwargs.get("batch_size"), shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=kwargs.get("batch_size"), shuffle=False)

        self.optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=1e-3)
        self.loss_fxn = torch.nn.functional.mse_loss

    def train(self):
        step = 0
        while True:
            self._train(step)
            self._test(step)
            self._sample(step)
            print()
            step += 1

    def _train(self, step: int):
        self.diffuser.train()
        total_loss = 0.0
        count = 0
        for x in tqdm(self.train_dataloader, desc=f"({wandb.run.name}) Train ({step})"):
            loss = self._train_inner(
                x.to(self.device),
            )

            total_loss += loss.item()
            count += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        wandb.log({"train_loss": total_loss / count}, step=step)

    def _test(self, step: int):
        self.diffuser.eval()
        total_loss = 0.0
        count = 0
        for x in tqdm(self.test_dataloader, desc=f"({wandb.run.name}) Test ({step})"):
            loss = self._train_inner(
                x.to(self.device),
            )
            total_loss += loss.item()
            count += 1

        wandb.log({"test_loss": total_loss / count}, step=step)

    def _train_inner(self, x: torch.Tensor):
        """
        (Original).
        1. Encode.
        2. Create noise. -> Noise.
        3. Add to latent and Predict noise. -> Predicted noise.
        4. Denoise.
        5. Decode. -> Decoded.
        Loss = MSE(Original., Decoded) + MSE(Noise, Predicted Noise)
        """
        latent = self.vae.encode(x)
        latent_noise = self.diffuser.create_noise(latent)
        noisy_latent = latent + latent_noise
        latent_noise_p = self.diffuser(latent + latent_noise)
        denoised_latent = noisy_latent - latent_noise_p
        x_p = self.vae.decode(denoised_latent)
        return self.loss_fxn(x_p, x) + self.loss_fxn(latent_noise, latent_noise_p) \
            + torch.nn.functional.binary_cross_entropy(x_p, x)

    def _sample(self, step: int):
        num_test_samples = len(self.test_dataset)
        random_test_sample_idx = random.randint(0, num_test_samples - 1)
        x = self.test_dataset[random_test_sample_idx].to(self.device)
        with torch.no_grad():
            latent = self.vae.encode(x)
            print("latent shape", latent.shape)
            latent_noise = self.diffuser.create_noise(latent, t=self.diffuser.total_timesteps * 2 // 3)
            noisy_latent = latent + latent_noise
            latent_noise_p = self.diffuser(latent + latent_noise)
            denoised_latent = noisy_latent - latent_noise_p
            x_p = self.vae.decode(denoised_latent)
        wandb.log({
            "original": self._prep_vid_for_wandb(x),
            "decoded": self._prep_vid_for_wandb(x_p),
        })
        print("Samples logged.")

    def _generate(self, step: int):
        with torch.no_grad():
            latent = torch.randn(1, 3, 16, 32, 32).to(self.device) # FIXME: shape
            for _ in tqdm(range(100), desc="Generating"):
                noise_p = self.diffuser(latent)
                latent = latent - noise_p
            x = self.vae.decode(latent)
        wandb.log({
            "generated": self._prep_vid_for_wandb(x),
        })
        print("Generation logged.")

    @staticmethod
    def _prep_vid_for_wandb(vid: torch.Tensor) -> torch.Tensor:
        vid = vid.detach().cpu()
        # clamp vid to [0, 1]
        vid = torch.clamp(vid, 0, 1)
        vid = rearrange(vid, "c t h w -> t c h w")
        vid = (vid * 255).to(torch.uint8)
        return wandb.Video(vid)


if __name__ == "__main__":
    device = "cuda:0"

    wandb.init(project="flying-mnist_tiny-diffuser")

    kwargs = {
        "device": "cuda:0",
        "batch_size": 32,
    }

    trainer = Trainer(**kwargs)
    trainer.train()

