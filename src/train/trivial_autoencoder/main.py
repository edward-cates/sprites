import argparse
from pathlib import Path

import wandb
import torch

from src.model.trivial_autoencoder import TrivialAutoencoder
from src.model.tiny_autoencoder import TinyAutoencoder
from src.dataset.sprites_dataset import SpritesDataset
from src.dataset.flying_mnist_dataset import FlyingMnistDataset
from src.train.trivial_autoencoder.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--gradient_accumulation", "-a", type=int, default=1)
    parser.add_argument("--learning_rate", "-r", type=float, default=1e-4)
    parser.add_argument("--gradient_clip", "-c", type=float, default=1.0)
    parser.add_argument("--checkpoint", "-m", type=str, default=None)
    parser.add_argument("--loss_type", "-l", type=str, default="mse,bce") # Can pass comma-separated to combine.
    args = parser.parse_args()

    assert all([
        loss in Trainer.losses.keys()
        for loss in args.loss_type.split(",")
    ]), f"Loss type must be one of {Trainer.losses.keys()}"

    model = TinyAutoencoder()
    if args.checkpoint is not None:
        assert Path(args.checkpoint).exists(), f"Checkpoint {args.checkpoint} not found"
        print(f"Loading checkpoint from {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        print("No checkpoint loaded, starting from scratch.")

    # dataset = SpritesDataset()
    # train_dataset, test_dataset = dataset.randomly_split(0.9)

    train_dataset = FlyingMnistDataset("train", max_samples=1000)
    test_dataset = FlyingMnistDataset("val", max_samples=100)

    wandb.init(project="flying-mnist_tiny-autoencoder_video_2")
    wandb.config.update(args)
    print(f"This run is named {wandb.run.name}.")

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        args=args,
    )

    trainer.train() # Infinite.

