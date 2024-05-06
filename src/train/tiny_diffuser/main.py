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
        for x in tqdm(self.train_dataloader):
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
        for x in tqdm(self.test_dataloader):
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
        x = self.test_dataset[random_test_sample_idx].to(self.device)
        noise = self.diffuser.create_noise(x)
        x_noisy = x + noise
        p_noise = self.diffuser(x_noisy)
        x_p = x_noisy - p_noise
        wandb.log({
            "original": self._prep_vid_for_wandb(x),
            "noisy": self._prep_vid_for_wandb(x_noisy),
            "predicted_noise": self._prep_vid_for_wandb(p_noise),
            "reconstructed": self._prep_vid_for_wandb(x_p),
        })

    @staticmethod
    def _prep_vid_for_wandb(vid: torch.Tensor) -> torch.Tensor:
        vid = torch.nn.Sigmoid()(vid)
        vid = rearrange(vid, "c t h w -> t c h w")
        vid = (vid * 255).to(torch.uint8)
        return vid


if __name__ == "__main__":
    device = "cuda:0"

    wandb.init(project="flying-mnist_tiny-diffuser")

    kwargs = {
        "device": "cuda:0",
        "batch_size": 32,
    }

    trainer = Trainer(**kwargs)
    trainer.train()

