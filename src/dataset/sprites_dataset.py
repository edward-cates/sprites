from typing import Optional, Tuple
from pathlib import Path

from PIL import Image
import torch
import torchvision

class SpritesDataset:
    def __init__(self, start_image: int = 0, end_image: Optional[int] = None):
        assert Path("data/sprites").exists(), "Sprites image folder not found - see README"
        self._start_image = start_image
        self._end_image = end_image if end_image is not None \
            else len(list(Path("data/sprites").rglob("*.png")))
        self._transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return self._end_image - self._start_image

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(f"data/sprites/{self._start_image + idx:04d}.png").convert("RGB")
        assert img.size == (64, 64), f"Image size is {img.size}, not (64, 64), for {self._start_image + idx:04d}.png"
        img_tensor = self._transforms(img)
        return img_tensor

    def split(self, fraction: float) -> Tuple["SpritesDataset", "SpritesDataset"]:
        assert 0.0 < fraction < 1.0, "Fraction must be between 0.0 and 1.0"
        split_idx = int(fraction * len(self))
        return SpritesDataset(self._start_image, self._start_image + split_idx), \
               SpritesDataset(self._start_image + split_idx, self._end_image)

    def get_random_image(self) -> torch.Tensor:
        idx = torch.randint(0, len(self), (1,)).item()
        return self[idx]
