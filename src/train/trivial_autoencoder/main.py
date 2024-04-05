import argparse
from pathlib import Path

import wandb
import torch

from src.model.trivial_autoencoder import TrivialAutoencoder
from src.model.tiny_autoencoder import TinyAutoencoder
from src.dataset.sprites_dataset import SpritesDataset
from src.train.trivial_autoencoder.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--learning_rate", "-r", type=float, default=1e-4)
    parser.add_argument("--gradient_clip", "-c", type=float, default=1.0)
    parser.add_argument("--checkpoint", "-m", type=str, default=None)
    parser.add_argument("--loss_type", "-l", type=str, default="mse") # Can pass comma-separated to combine.
    args = parser.parse_args()

    assert all([
        loss in Trainer.losses.keys()
        for loss in args.loss_type.split(",")
    ]), f"Loss type must be one of {Trainer.losses.keys()}"

    wandb.init(project="sprites_trivial-autoencoder")
    wandb.config.update(args)

    print(f"This run is named {wandb.run.name}.")

    model = TinyAutoencoder()
    if args.checkpoint is not None:
        assert Path(args.checkpoint).exists(), f"Checkpoint {args.checkpoint} not found"
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        print("No checkpoint loaded, starting from scratch.")
    dataset = SpritesDataset()
    trainer = Trainer(model=model, dataset=dataset, args=args)

    trainer.train() # Infinite.
