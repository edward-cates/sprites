import random
from argparse import Namespace
from typing import Callable, Iterable
from pathlib import Path

import wandb
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from tqdm import tqdm
from einops import rearrange

from src.dataset.sprites_dataset import SpritesDataset
from src.loss.vgg_perceptual_loss import VGGPerceptualLoss

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataset: Iterable,
            test_dataset: Iterable,
            args: Namespace,
    ):
        # model = torch.nn.DataParallel(model)
        self.model = model.to(args.device)
        self.args = args

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        if "ssim" in self.args.loss_type:
            self._initialize_ssim()
        if "perc" in self.args.loss_type:
            self._initialize_perc_loss()

    def train(self):
        # Sanity check the data.
        self._log_sample_data()

        step = 0
        while True:
            self._train(step=step)
            self._test(step=step)
            wandb.run.save()
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
        ssim = SSIM(
            data_range=1.0,
            kernel_size=5,
        ).to(self.args.device)
        self.losses["ssim"] = lambda output, target: 1.0 - ssim(output, target)

    def _initialize_perc_loss(self) -> Callable:
        vgg_loss = VGGPerceptualLoss().to(self.args.device)
        self.losses["perc"] = lambda output, target: vgg_loss(output, target)

    def _train(self, step: int) -> None:
        self.model.train()
        total_loss = 0.0
        count = 0

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="Training",
        )
        for batch_idx, batch in pbar:
            x = batch.to(self.args.device)

            y, mu, logvar = self.model(
                # Random erasing.
                torch.stack([
                    self._remove_random_frames(vid)
                    for vid in x
                ])
            )

            loss = self._calc_loss(batch_idx=batch_idx, output=y, mu=mu, logvar=logvar, target=x)
            loss = loss / self.args.gradient_accumulation
            loss.backward()

            if (((batch_idx + 1) % self.args.gradient_accumulation) == 0) \
                    or (batch_idx == (len(self.train_loader) - 1)):
                # clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()

            pbar.set_postfix({"Loss": loss.item(), f"({wandb.run.name}) Train Step": step})
            total_loss += loss.item()
            count += 1

        print(f"Train loss epoch: {total_loss / count}")
        wandb.log({"train_loss_epoch": total_loss / count}, step=step)

    def _test(self, step: int) -> None:
        self.model.eval()
        total_loss = 0.0
        count = 0

        pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc="Testing")
        with torch.no_grad():
            for batch_idx, batch in pbar:
                x = batch.to(self.args.device)
                y, mu, logvar = self.model(
                    torch.stack([
                        self._remove_random_frames(vid)
                        for vid in x
                    ])
                )
                loss = self._calc_loss(batch_idx=batch_idx, output=y, mu=mu, logvar=logvar, target=x)
                loss = loss / self.args.gradient_accumulation

                pbar.set_postfix({"Loss": loss.item(), f"({wandb.run.name}) Test Step": step})
                total_loss += loss.item()
                count += 1

            print(f"Test loss epoch: {total_loss / count}")
            wandb.log({"test_loss_epoch": total_loss / count}, step=step)

            if step % 30 == 0 or True:
                # Pick 3 random images from the test set and log the input and output.
                x = Trainer._get_random_video_from_dataloader(self.test_loader).unsqueeze(0).to(self.args.device)
                x = torch.stack([
                    self._remove_random_frames(vid)
                    for vid in x
                ])
                y, _, _ = self.model(x)

                assert x.shape == y.shape, f"x shape ({x.shape}) != y shape ({y.shape})"

                wandb.log({
                    "input_images_2": [wandb.Video(self._prep_vid_for_wandb(img)) for img in x.cpu()],
                    "output_images": [wandb.Video(self._prep_vid_for_wandb(img)) for img in y.cpu()],
                }, step=step)

                # Save checkpoint to disk.
                checkpoint_dir = Path("checkpoints") / wandb.run.project / wandb.run.name
                checkpoint_dir.mkdir(exist_ok=True, parents=True)
                torch.save(self.model.state_dict(), str(checkpoint_dir / f"checkpoint_at_step_{step}.pth"))

    def _calc_loss(self, batch_idx: int, output: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        frame_losses = sum([
            self._calc_frame_loss(output[:, :, i], target[:, :, i])
            for i in range(target.shape[1])
        ])
        transition_losses = sum([
            self._calc_transition_loss(
                output[:, :, i + 1] - output[:, :, i],
                target[:, :, i + 1] - target[:, :, i],
            )
            for i in range(7)
        ])
        # kl_divergence = torch.sqrt(self._calc_kl_divergence(mu, logvar))

        if batch_idx == 0:
            print(f"Frame loss: {frame_losses}, Transition loss: {transition_losses}")#, KL Divergence: {kl_divergence}")

        return frame_losses + transition_losses# + kl_divergence

    def _calc_kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def _calc_frame_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sum([
            self.losses[loss](output, target)
            for loss in self.args.loss_type.split(",")
        ])

    def _calc_transition_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.losses["mse"](output, target)

    @staticmethod
    def _prep_vid_for_wandb(vid: torch.Tensor) -> torch.Tensor:
        vid = rearrange(vid, "c t h w -> t c h w")
        vid = (vid * 255).to(torch.uint8)
        return vid

    @staticmethod
    def _remove_random_frames(vid: torch.Tensor) -> torch.Tensor:
        # t is second dimension.
        vid = vid.clone()
        t = vid.shape[1]
        last_frame_removed = -1
        for _ in range(t // 2):
            first_frame_available = last_frame_removed + 1
            last_frame_available = t - 1
            if first_frame_available > last_frame_available:
                break
            # `randint` includes both endpoints.
            frame_idx = random.randint(first_frame_available, last_frame_available)
            # copy vid and set frame to random noise.
            # vid[:, frame_idx] = torch.randn_like(vid[:, frame_idx])
            vid[:, frame_idx] = torch.zeros_like(vid[:, frame_idx])
        return vid

    @staticmethod
    def _get_random_video_from_dataloader(dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        dataset = dataloader.dataset
        size = len(dataset)
        idx = random.randint(0, size - 1)
        return dataset[idx]

    def _log_sample_data(self) -> None:
        # Pick 3 random images from the test set and log the input and output.
        x = torch.stack([
            Trainer._get_random_video_from_dataloader(self.train_loader)
            for _ in range(2)
        ])

        wandb.log({
            "data_check": [wandb.Video(self._prep_vid_for_wandb(img)) for img in x.cpu()],
        }, step=0)

        print("Logged data sample.")

