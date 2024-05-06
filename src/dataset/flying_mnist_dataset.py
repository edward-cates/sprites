from pathlib import Path

import torch
import torchvision
from einops import rearrange

class FlyingMnistDataset:
    def __init__(self, train_or_val: str):
        assert train_or_val in ["train", "val"], \
            f"train_or_val must be 'train' or 'val', not {train_or_val}."
        self.is_val = train_or_val == "val"
        self.dir = Path(f"data/flying-mnist/flying_mnist_11k/{train_or_val}")
        assert self.dir.exists(), f"Directory {self.dir} does not exist."
        self.num_samples = len(list(self.dir.glob("*.mp4")))

    def __len__(self):
        min_ = 100 if self.is_val else 1000
        return min(min_, self.num_samples)

    def __getitem__(self, idx) -> torch.Tensor:
        return self._preprocess_video(
            video_data=self._get_video_data(idx),
        )

    # Private.

    def _get_video_path(self, idx) -> Path:
        return self.dir / f"{idx:05d}.mp4"

    def _get_video_data(self, idx) -> torch.Tensor:
        video_path = self._get_video_path(idx)
        video_data, audio_data, metadata = torchvision.io.read_video(str(video_path))
        return video_data

    @staticmethod
    def _preprocess_video(video_data: torch.Tensor) -> torch.Tensor:
        # choose a random frame and change to all zeros.
        assert video_data.ndim == 4, f"video_data must have 4 dimensions, not {video_data.ndim}."
        assert video_data.shape[1:] == (512, 512, 3), \
            f"video_data must have shape (T, 512, 512, 3), not {video_data.shape}."
        video_data = FlyingMnistDataset._keep_first_second(video_data)
        video_data = FlyingMnistDataset._normalize_video(video_data)
        video_data = rearrange(video_data, "t h w c -> c t h w")
        video_data = FlyingMnistDataset._resize_video(video_data)
        return video_data

    @staticmethod
    def _keep_first_second(video_data: torch.Tensor) -> torch.Tensor:
        return video_data[:8]

    @staticmethod
    def _resize_video(video_data: torch.Tensor) -> torch.Tensor:
        img_size = 64
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size)),
        ])
        return transforms(video_data)

    @staticmethod
    def _normalize_video(video_data: torch.Tensor) -> torch.Tensor:
        return video_data.float() / 255.0

