import random

from tqdm import tqdm
import wandb
import torch
from einops import rearrange

from src.model.tiny_diffuser import TinyDiffuser

from src.dataset.flying_mnist_dataset import FlyingMnistDataset

class Trainer:
    def __init__(self, **kwargs):
        self.device = kwargs.get("device")

        self.diffuser = TinyDiffuser()
        self.diffuser.to(device)

        self.train_dataset = FlyingMnistDataset("train")
        self.test_dataset = FlyingMnistDataset("val")

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=kwargs.get("batch_size"), shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=kwargs.get("batch_size"), shuffle=False)

        self.optimizer = torch.optim.Adam(self.diffuser.parameters(), lr=1e-4)
        self.loss_fxn = lambda outputs, targets: torch.nn.functional.mse_loss(outputs, targets)

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
            # save checkpoint
            torch.save(self.diffuser.state_dict(), f"checkpoints/diffusion/tiny_diffuser_{step}.pt")

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
        noise = self.diffuser.create_noise(x)
        p_noise = self.diffuser(x + noise)
        return self.loss_fxn(p_noise, noise)

    def _sample(self, step: int):
        num_test_samples = len(self.test_dataset)
        random_test_sample_idx = random.randint(0, num_test_samples - 1)
        x = self.test_dataset[random_test_sample_idx].unsqueeze(0).to(self.device)
        with torch.no_grad():
            noise = self.diffuser.create_noise(x, t=self.diffuser.total_timesteps * 2 // 3)
            x_noisy = x + noise
            p_noise = self.diffuser(x_noisy)
            x_p = x_noisy - p_noise
        wandb.log({
            "original": self._prep_vid_for_wandb(x[0]),
            "noisy": self._prep_vid_for_wandb(x_noisy[0]),
            "predicted_noise": self._prep_vid_for_wandb(p_noise[0]),
            "reconstructed": self._prep_vid_for_wandb(x_p[0]),
        }, step=step)
        print("Samples logged.")

    def _generate(self, step: int):
        with torch.no_grad():
            x = torch.randn(1, 3, 8, 64, 64).to(self.device)
            intermediates = [x[0].clone()]
            for ix in tqdm(range(100), desc="Generating"):
                noise_p = self.diffuser(x)
                x_t = x - noise_p
                x = 0.9 * x + 0.1 * x_t
                if (ix + 1) % 25 == 0:
                    intermediates.append(x[0].clone())
        wandb.log({
            "generated": [
                self._prep_vid_for_wandb(intermediate)
                for intermediate in intermediates
            ],
        }, step=step)
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

