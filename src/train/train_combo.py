import random
from typing import Optional
from pathlib import Path

from tqdm import tqdm
import wandb
import torch
from einops import rearrange

from src.model.tiny_diffuser import TinyDiffuser
from src.model.tiny_autoencoder import TinyAutoencoder

from src.dataset.flying_mnist_dataset import FlyingMnistDataset
from src.dataset.sprites_dataset import SpritesDataset
from src.dataset.virat_dataset import ViratDataset
from src.dataset.pexels_dataset import PexelsDataset
from src.dataset.fingers_dataset import FingersDataset
class Trainer:
    def __init__(self, **kwargs):
        self.device = kwargs.get("device")

        # previous_checkpoint_dir = Path("checkpoints/flying-mnist_tiny-combo-2/golden-butterfly-31")
        # previous_checkpoint_step = 400

        self.vae = TinyAutoencoder()
        # self.vae.load_state_dict(torch.load(str(previous_checkpoint_dir / f"vae_at_step_{previous_checkpoint_step}.pth")))
        self.vae.to(self.device)

        self.diffuser = TinyDiffuser(in_channels=128)
        # self.diffuser.load_state_dict(torch.load(str(previous_checkpoint_dir / f"diffuser_at_step_{previous_checkpoint_step}.pth")))
        self.diffuser.to(self.device)

        # if kwargs.get("use_sprites") is not True:
        #     self.train_dataset = FlyingMnistDataset("train")#, max_samples=1000)
        #     self.test_dataset = FlyingMnistDataset("val")#, max_samples=100)
        # else:
        #     sprites_dataset = SpritesDataset()
        #     self.train_dataset, self.test_dataset = sprites_dataset.randomly_split(0.9)

        # dataset = ViratDataset.from_tiny_virat(max_samples=7000)
        # dataset = PexelsDataset.from_tiny_virat(max_samples=7000)
        dataset = FingersDataset.from_tiny_virat(max_samples=7000)
        self.train_dataset, self.test_dataset = dataset, dataset

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=kwargs.get("batch_size"), shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=kwargs.get("batch_size"), shuffle=False)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.vae.parameters()},
                {"params": self.diffuser.parameters()},
            ],
            lr=1e-5,
            weight_decay=1e-5,
        )

        self.latent_shape = None

    def train(self):
        step = 0
        while True:
            self._train(step)
            self._test(step)
            if step % 10 == 0:
                self._sample(step)
                self._generate(step)
                self._save_models(step)
                wandb.run.save()
            print()
            step += 1

    def _train(self, step: int):
        self.diffuser.train()
        self.vae.train()
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

            # loss = image_loss + noise_loss
            loss = (1 + torch.log(1 + image_loss)) * (1 + torch.log(1 + noise_loss))
            total_loss += loss.item()
            count += 1

            loss.backward()
            # clip gradients.
            torch.nn.utils.clip_grad_norm_(self.vae.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.diffuser.parameters(), 0.5)
            self.optimizer.step()
            self.optimizer.zero_grad()

        wandb.log({"train_loss": total_loss / count}, step=step)
        wandb.log({"train_image_loss": total_image_loss / count}, step=step)
        wandb.log({"train_noise_loss": total_noise_loss / count}, step=step)

    def _test(self, step: int):
        self.diffuser.eval()
        self.vae.eval()
        total_loss = 0.0
        total_image_loss = 0
        total_noise_loss = 0
        count = 0
        pbar = tqdm(self.test_dataloader, desc=f"({wandb.run.name}) Test ({step})")
        with torch.no_grad():
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
        # latent = self.vae.encode(
        #     # Random erasing.
        #     torch.stack([
        #         self._remove_random_frame(vid)
        #         for vid in x
        #     ]),
        # )
        # don't do random erasing.
        latent = self.vae.encode(x)
        latent_noisy, t, noise = self.diffuser.create_noised_image(latent)
        predicted_noise = self.diffuser(latent_noisy)
        actual_noise = latent_noisy - latent
        latent_0 = latent_noisy - predicted_noise
        x_0 = self.vae.decode(latent_0)
        x_ae = self.vae.decode(latent)
        ae_loss = torch.nn.functional.mse_loss(x_ae, x, reduction='mean')
        ae_transition_loss = self._calc_transition_loss(x_ae, x)
        img_loss = torch.nn.functional.mse_loss(x_0, x, reduction='mean')
        img_transition_loss = self._calc_transition_loss(x_0, x)
        # 100 is observed to be a good scaling factor.
        img_loss = (img_loss + img_transition_loss + ae_loss + ae_transition_loss) / 4.0 * 50.0
        noise_loss = torch.nn.functional.mse_loss(predicted_noise, actual_noise, reduction='mean')
        pbar.set_postfix({"Image Loss": img_loss.item(), "Noise Loss": noise_loss.item()})
        return img_loss, noise_loss

    @staticmethod
    def _calc_transition_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = Trainer._calc_transitions(x)
        y = Trainer._calc_transitions(y)
        return torch.nn.functional.mse_loss(x, y, reduction='mean')

    @staticmethod
    def _calc_transitions(x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:] - x[:, :-1]

    def _sample(self, step: int):
        num_test_samples = len(self.test_dataset)
        random_test_sample_idx = random.randint(0, num_test_samples - 1)
        x = self.test_dataset[random_test_sample_idx].unsqueeze(0).to(self.device)
        with torch.no_grad():
            latent = self.vae.encode(x)
            print("latent shape:", latent.shape)
            # assert latent.shape == (1, 64, 8, 4, 4)
            if self.latent_shape is None:
                self.latent_shape = latent.shape
            t = 100
            latent_noisy, t_b, noise = self.diffuser.create_noised_image(latent, t=t)
            predicted_noise = self.diffuser(latent_noisy)
            predicted_latent_0 = latent_noisy - predicted_noise
            x_p = self.vae.decode(predicted_latent_0)
        wandb.log({
            "original": self._prep_vid_for_wandb(x[0]),
            "decoded": self._prep_vid_for_wandb(x_p[0]),
        }, step=step)
        print("Samples logged.")

    def _generate(self, step: int):
        with torch.no_grad():
            latent = self.diffuser.generate(
                # torch.randn(1, 64, 8, 4, 4).to(self.device),
                torch.randn(1, *self.latent_shape[1:]).to(self.device),
            )
            intermediates = [
                self.vae.decode(latent)[0].clone()
            ]
        wandb.log({
            "generated": [
                self._prep_vid_for_wandb(intermediate)
                for intermediate in intermediates
            ],
        }, step=step)
        print("Generation logged.")

    @staticmethod
    def _remove_random_frame(vid: torch.Tensor) -> torch.Tensor:
        return vid
        # t is second dimension.
        t = vid.shape[1]
        frame_idx = random.randint(0, t - 1)
        # copy vid and set frame to zeros.
        vid = vid.clone()
        # random
        vid[:, frame_idx] = torch.randn_like(vid[:, frame_idx])
        return vid

    @staticmethod
    def _prep_vid_for_wandb(vid: torch.Tensor) -> torch.Tensor:
        vid = vid.detach().cpu()
        # clamp vid to [0, 1]
        vid = torch.clamp(vid, 0, 1)
        vid = rearrange(vid, "c t h w -> t c h w")
        vid = (vid * 255).to(torch.uint8)
        return wandb.Video(vid)

    def _save_models(self, step: int):
        # Save checkpoint to disk.
        checkpoint_dir = Path("checkpoints") / wandb.run.project / wandb.run.name
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        torch.save(self.diffuser.state_dict(), str(checkpoint_dir / f"diffuser_at_step_{step}.pth"))
        torch.save(self.vae.state_dict(), str(checkpoint_dir / f"vae_at_step_{step}.pth"))


if __name__ == "__main__":
    wandb.init(project="fingers")

    kwargs = {
        "device": "cuda:0",
        "batch_size": 8,
        "use_sprites": False,
    }

    trainer = Trainer(**kwargs)
    trainer.train()

