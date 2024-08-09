from typing import Optional, Tuple, List
from pathlib import Path
import random
from einops import rearrange

from PIL import Image
import torch
import torchvision

class ViratDataset:
    def __init__(self, video_paths: List[Path]):
        self.video_paths = video_paths
        self._image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    @classmethod
    def from_tiny_virat(cls, max_samples: int = -1):
        assert Path("data/TinyVIRAT").exists()
        train_dir = Path("data/TinyVIRAT/videos/train")
        test_dir = Path("data/TinyVIRAT/videos/test")
        video_paths = list(train_dir.glob("*/*.mp4")) \
            + list(test_dir.glob("*/*.mp4"))
        # shuffle.
        random.shuffle(video_paths)
        if max_samples > 0:
            video_paths = video_paths[:max_samples]
        return cls(video_paths)

    def split(self, fraction: float) -> Tuple["ViratDataset", "ViratDataset"]:
        split_idx = int(len(self) * fraction)
        return ViratDataset(self.video_paths[:split_idx]), \
               ViratDataset(self.video_paths[split_idx:])

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._preprocess_video(
            video_data=self._get_video_data(idx),
        )

    def _get_video_data(self, idx) -> torch.Tensor:
        video_path = self.video_paths[idx]
        video_data, audio_data, metadata = torchvision.io.read_video(str(video_path))
        return video_data

    @staticmethod
    def _preprocess_video(video_data: torch.Tensor) -> torch.Tensor:
        video_data = ViratDataset._keep_first_second(video_data)
        video_data = ViratDataset._normalize_video(video_data)
        video_data = rearrange(video_data, "t h w c -> c t h w")
        video_data = ViratDataset._resize_video(video_data)
        return video_data

    @staticmethod
    def _keep_first_second(video_data: torch.Tensor) -> torch.Tensor:
        return video_data[::2][:16]

    @staticmethod
    def _resize_video(video_data: torch.Tensor) -> torch.Tensor:
        img_size = 128
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((img_size, img_size)),
        ])
        return transforms(video_data)

    @staticmethod
    def _normalize_video(video_data: torch.Tensor) -> torch.Tensor:
        return video_data.float() / 255.0

