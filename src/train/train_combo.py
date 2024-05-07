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
        self.vae.to(self.device)

        self.diffuser = TinyDiffuser(in_channels=64)
        self.diffuser.to(self.device)

        self.train_dataset = FlyingMnistDataset("train", max_samples=1000)
        self.test_dataset = FlyingMnistDataset("val", max_samples=100)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=kwargs.get("batch_size"), shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=kwargs.get("batch_size"), shuffle=False)

        self.optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=3e-5)
        self.loss_fxn = torch.nn.functional.mse_loss

    def train(self):
        step = 0
        while True:
            self._train(step)
            self._test(step)
            self._sample(step)
            self._generate(step)
            wandb.run.save()
            print()
            step += 1

    def _train(self, step: int):
        self.diffuser.train()
        total_loss = 0.0
        total_image_loss = 0
        total_noise_loss = 0
        count = 0
        pbar = tqdm(self.train_dataloader, desc=f"({wandb.run.name}) Train ({step})")
        for x in pbar:
            image_loss, noise_loss = self._train_inner(
                x.to(self.device),
                pbar=pbar,
            )

            total_image_loss += image_loss.item()
            total_noise_loss += noise_loss.item()

            loss = image_loss + noise_loss
            total_loss += loss.item()
            count += 1

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        wandb.log({"train_loss": total_loss / count}, step=step)
        wandb.log({"train_image_loss": total_image_loss / count}, step=step)
        wandb.log({"train_noise_loss": total_noise_loss / count}, step=step)

    def _test(self, step: int):
        self.diffuser.eval()
        total_loss = 0.0
        total_image_loss = 0
        total_noise_loss = 0
        count = 0
        pbar = tqdm(self.test_dataloader, desc=f"({wandb.run.name}) Test ({step})")
        for x in pbar:
            image_loss, noise_loss = self._train_inner(
                x.to(self.device),
                pbar=pbar,
            )
            total_image_loss += image_loss.item()
            total_noise_loss += noise_loss.item()

            loss = image_loss + noise_loss
            total_loss += loss.item()

            count += 1

        wandb.log({"test_loss": total_loss / count}, step=step)
        wandb.log({"test_image_loss": total_image_loss / count}, step=step)
        wandb.log({"test_noise_loss": total_noise_loss / count}, step=step)

    def _train_inner(self, x: torch.Tensor, pbar):
        """
        (Original).
        1. Encode.
        2. Create noise. -> Noise.
        3. Add to latent and Predict noise. -> Predicted noise.
        4. Denoise.
        5. Decode. -> Decoded.
        Loss = MSE(Original., Decoded) + MSE(Noise, Predicted Noise)
        """
        latent = self.vae.encode(
            # Random erasing.
            torch.stack([
                self._remove_random_frame(vid)
                for vid in x
            ]),
        )
        latent_noise = self.diffuser.create_noise(latent)
        noisy_latent = latent + latent_noise
        latent_noise_p = self.diffuser(latent + latent_noise)
        denoised_latent = noisy_latent - latent_noise_p
        x_p = self.vae.decode(denoised_latent)
        img_loss = self.loss_fxn(x_p, x)
        noise_loss = self.loss_fxn(latent_noise, latent_noise_p)
        pbar.set_postfix({"Image Loss": img_loss.item(), "Noise Loss": noise_loss.item()})
        return img_loss, noise_loss

    def _sample(self, step: int):
        num_test_samples = len(self.test_dataset)
        random_test_sample_idx = random.randint(0, num_test_samples - 1)
        x = self.test_dataset[random_test_sample_idx].unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.vae.encode(x)
            print("latent shape:", latent.shape)
            latent_noise = self.diffuser.create_noise(latent, t=self.diffuser.total_timesteps * 2 // 3)
            noisy_latent = latent + latent_noise
            latent_noise_p = self.diffuser(latent + latent_noise)
            denoised_latent = noisy_latent - latent_noise_p
            x_p = self.vae.decode(denoised_latent)
        wandb.log({
            "original": self._prep_vid_for_wandb(x[0]),
            "noisy": self._prep_vid_for_wandb(self.vae.decode(noisy_latent)[0]),
            "predicted_noise": self._prep_vid_for_wandb(self.vae.decode(latent_noise_p)[0]),
            "decoded": self._prep_vid_for_wandb(x_p[0]),
        }, step=step)
        print("Samples logged.")

    def _generate(self, step: int):
        with torch.no_grad():
            latent = torch.randn(1, 64, 4, 32, 32).to(self.device)
            intermediates = [
                self.vae.decode(latent)[0].clone()
            ]
            for ix in tqdm(range(100), desc="Generating"):
                noise_p = self.diffuser(latent)
                latent_t = latent - noise_p
                latent = 0.9 * latent + 0.1 * latent_t
                if (ix + 1) % 25 == 0:
                    intermediates.append(
                        self.vae.decode(latent)[0].clone(),
                    )
        wandb.log({
            "generated_2": [
                self._prep_vid_for_wandb(intermediate)
                for intermediate in intermediates
            ],
        }, step=step)
        print("Generation logged.")

    @staticmethod
    def _remove_random_frame(vid: torch.Tensor) -> torch.Tensor:
        # t is second dimension.
        t = vid.shape[1]
        frame_idx = random.randint(0, t - 1)
        # copy vid and set frame to zeros.
        vid = vid.clone()
        vid[:, frame_idx] = 0
        return vid

    @staticmethod
    def _prep_vid_for_wandb(vid: torch.Tensor) -> torch.Tensor:
        vid = vid.detach().cpu()
        # clamp vid to [0, 1]
        vid = torch.clamp(vid, 0, 1)
        vid = rearrange(vid, "c t h w -> t c h w")
        vid = (vid * 255).to(torch.uint8)
        return wandb.Video(vid)


if __name__ == "__main__":
    wandb.init(project="flying-mnist_tiny-combo")

    kwargs = {
        "device": "cuda:1",
        "batch_size": 32,
    }

    trainer = Trainer(**kwargs)
    trainer.train()

