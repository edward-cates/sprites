import argparse
import random
from pathlib import Path

from tqdm import tqdm
import wandb
import torch
from einops import rearrange

from src.model.tiny_diffuser import TinyDiffuser

from src.dataset.flying_mnist_dataset import FlyingMnistDataset
from src.dataset.sprites_dataset import SpritesDataset

class Trainer:
    def __init__(self, **kwargs):
        self.device = kwargs.get("device")
        self.kwargs = kwargs

        self.diffuser = TinyDiffuser()
        if kwargs.get("checkpoint") is not None:
            self.diffuser.load_state_dict(torch.load(kwargs["checkpoint"]))
        self.diffuser.to(self.device)

        if kwargs.get("use_sprites") is not True:
            self.train_dataset = FlyingMnistDataset("train", max_samples=4000)
            self.test_dataset = FlyingMnistDataset("val", max_samples=400)
        else:
            sprites_dataset = SpritesDataset()
            self.train_dataset, self.test_dataset = sprites_dataset.randomly_split(0.9)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=kwargs.get("batch_size"), shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=kwargs.get("batch_size"), shuffle=False)

        self.optimizer = torch.optim.AdamW(self.diffuser.parameters(), lr=7e-6)

    def train(self):
        step = 0
        while True:
            self._train(step)
            self._test(step)
            if (self.kwargs.get("use_sprites") is not True) or (step % 20 == 0):
                self._sample(step)
                self._generate(step)
                self._save_checkpoint(step)
            wandb.run.save()
            print()
            step += 1
            # save checkpoint

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
        print(f"Train loss: {total_loss / count}")

    def _save_checkpoint(self, step: int):
        checkpoint_dir = Path("checkpoints") / wandb.run.project / wandb.run.name
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        torch.save(self.diffuser.state_dict(), str(checkpoint_dir / f"checkpoint_at_step_{step}.pth"))

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
        print(f"Test loss: {total_loss / count}")

    def _train_inner(self, x: torch.Tensor):
        x_noisy, t, noise = self.diffuser.create_noised_image(x)
        predicted_noise = self.diffuser(x_noisy)
        # x_pred = self.diffuser.noise_scheduler.predict_start_from_noise(
        #     x_t=x_noisy,
        #     t=t,
        #     noise=predicted_noise,
        # )
        # return torch.nn.functional.mse_loss(x, x_pred)
        return torch.nn.functional.mse_loss(predicted_noise, x_noisy - x)

    def _sample(self, step: int):
        num_test_samples = len(self.test_dataset)
        random_test_sample_idx = random.randint(0, num_test_samples - 1)
        x = self.test_dataset[random_test_sample_idx].unsqueeze(0).to(self.device)
        with torch.no_grad():
            t = 100
            x_noisy, t_b, noise = self.diffuser.create_noised_image(x, t=t)
            predicted_noise = self.diffuser(x_noisy)
            predicted_x_0 = x_noisy - predicted_noise
            # predicted_x_0 = self.diffuser.noise_scheduler.predict_start_from_noise(
            #     x_t=x_noisy,
            #     t=t_b,
            #     noise=predicted_noise,
            # )
        wandb.log({
            "original": self._prep_vid_for_wandb(x[0]),
            "noisy": self._prep_vid_for_wandb(x_noisy[0]),
            "predicted_noise": self._prep_vid_for_wandb(predicted_noise[0]),
            "reconstructed": self._prep_vid_for_wandb(predicted_x_0[0]),
        }, step=step)
        print("Samples logged.")

    def _generate(self, step: int):
        """
        create a random tensor x.
        for each timestep:
          langevein diffusion step.
          -> add a random perturbation (according to the noise schedule?)
        """
        # https://www.assemblyai.com/blog/minimagen-build-your-own-imagen-text-to-image-model/
        example_x = self.test_dataset[0].unsqueeze(0).to(self.device)
        x = self.diffuser.generate(example_x)
        wandb.log({
            "generated": [
                self._prep_vid_for_wandb(x[0])
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
    # Args:
    parser = argparse.ArgumentParser()
    # checkpoint arg:
    parser.add_argument("--checkpoint", "-m", type=str, default=None)
    args = parser.parse_args()

    wandb.init(project="flying-mnist_tiny-diffuser-Aai-1")

    kwargs = {
        "device": "cuda:0",
        "batch_size": 32,
        "checkpoint": args.checkpoint,
        "use_sprites": True,
    }
    wandb.config.update(kwargs)

    trainer = Trainer(**kwargs)
    trainer.train()

