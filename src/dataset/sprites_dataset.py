from typing import Optional, Tuple, List
from pathlib import Path
import random
from einops import rearrange

from PIL import Image
import torch
import torchvision

class SpritesDataset:
    def __init__(self, image_indexes: Optional[List[int]] = None):
        assert Path("data/sprites").exists(), "Sprites image folder not found - see README"
        self._image_indexes = sorted(image_indexes) if image_indexes is not None \
            else list(range(912))
        self._image_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

    def randomly_split(self, fraction: float) -> Tuple["SpritesDataset", "SpritesDataset"]:
        random.shuffle(self._image_indexes)
        split_idx = int(len(self) * fraction)
        return SpritesDataset(self._image_indexes[:split_idx]), \
               SpritesDataset(self._image_indexes[split_idx:])

    def __len__(self) -> int:
        return len(self._image_indexes)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_idx = self._image_indexes[idx]
        left_photo = self._image_transforms(
            Image.open(f"data/retro-pixel-characters-generator/data/data/1/{img_idx}.png").convert("RGB"),
        )
        center_photo = self._image_transforms(
            Image.open(f"data/retro-pixel-characters-generator/data/data/2/{img_idx}.png").convert("RGB"),
        )
        right_photo = self._image_transforms(
            Image.open(f"data/retro-pixel-characters-generator/data/data/3/{img_idx}.png").convert("RGB"),
        )
        # Make T=8 to fit properly into the autoencoder.
        video = torch.stack([
            left_photo,
            left_photo,
            center_photo,
            center_photo,
            right_photo,
            right_photo,
            center_photo,
            center_photo,
        ])
        return rearrange(video, "t c h w -> c t h w")

    def get_random_image(self) -> torch.Tensor:
        idx = torch.randint(0, len(self), (1,)).item()
        return self[idx]
