from argparse import Namespace
from typing import Callable
from pathlib import Path

import wandb
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm

from src.dataset.sprites_dataset import SpritesDataset
from src.loss.vgg_perceptual_loss import VGGPerceptualLoss

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            dataset: SpritesDataset,
            args: Namespace,
    ):
        self.model = model.to(args.device)
        self.args = args

        train_set, test_set = dataset.split(0.8)
        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        if "ssim" in self.args.loss_type:
            self._initialize_ssim()
        if "perc" in self.args.loss_type:
            self._initialize_perc_loss()

    def train(self):
        step = 0
        while True:
            self._train(step=step)
            self._test(step=step)
            print()
            step += 1

    losses = {
        "mse": torch.nn.functional.mse_loss,
        "bce": torch.nn.functional.binary_cross_entropy,
        "ssim": None,
        "perc": None,
    }

    # Private

    def _initialize_ssim(self) -> Callable:
        ssim = SSIM(data_range=1.0).to(self.args.device)
        self.losses["ssim"] = lambda output, target: 1.0 - ssim(output, target)

    def _initialize_perc_loss(self) -> Callable:
        vgg_loss = VGGPerceptualLoss().to(self.args.device)
        self.losses["perc"] = lambda output, target: vgg_loss(output, target)

    def _train(self, step: int) -> None:
        self.model.train()
        total_loss = 0.0
        count = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            self.optimizer.zero_grad()

            x = batch.to(self.args.device)
            y = self.model(x)
            loss = self._calc_loss(output=y, target=x)
            loss.backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
            self.optimizer.step()

            pbar.set_postfix({"Loss": loss.item(), f"{wandb.run.name} Train Step": step})
            total_loss += loss.item()
            count += 1
        
        print(f"Train loss epoch: {total_loss / count}")
        wandb.log({"train_loss_epoch": total_loss / count}, step=step)

    def _test(self, step: int) -> None:
        self.model.eval()
        total_loss = 0.0
        count = 0

        pbar = tqdm(self.test_loader, desc="Testing")
        with torch.no_grad():
            for batch in pbar:
                x = batch.to(self.args.device)
                y = self.model(x)
                loss = self._calc_loss(output=y, target=x)

                pbar.set_postfix({"Loss": loss.item(), f"{wandb.run.name} Test Step": step})
                total_loss += loss.item()
                count += 1

        print(f"Test loss epoch: {total_loss / count}")
        wandb.log({"test_loss_epoch": total_loss / count}, step=step)

        if step % 10 == 0:
            # Pick 3 random images from the test set and log the input and output.
            x = torch.stack([
                self.test_loader.dataset.get_random_image()
                for _ in range(3)
            ]).to(self.args.device)
            y = self.model(x)

            wandb.log({
                "input_images": [wandb.Image(img) for img in x.cpu()],
                "output_images": [wandb.Image(img) for img in y.cpu()],
            }, step=step)

            # Save checkpoint to disk.
            checkpoint_dir = Path("checkpoints") / wandb.run.name
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            torch.save(self.model.state_dict(), str(checkpoint_dir / f"checkpoint_at_step_{step}.pth"))

    def _calc_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sum([
            self.losses[loss](output, target)
            for loss in self.args.loss_type.split(",")
        ])
