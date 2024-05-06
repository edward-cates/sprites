# https://github.com/madebyollin/taesd/blob/main/taesd.py

import torch
import torch.nn as nn

def conv(n_in, n_out, **kwargs):
    return nn.Conv3d(n_in, n_out, 3, padding=1, **kwargs)

class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3

class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv3d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))

def Encoder():
    return nn.Sequential(
        conv(3, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 4),
    )

def Decoder():
    return nn.Sequential(
        Clamp(), conv(4, 64), nn.ReLU(),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )

class IdentityConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.batch_norm = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        y = self.conv(x)
        y = self.batch_norm(y)
        y = self.relu(y)
        return y

class IdentityDeconv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.ConvTranspose3d(64, 3, kernel_size=6, stride=2, padding=2, bias=False)
        self.batch_norm = nn.BatchNorm3d(3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(3, 3, kernel_size=1, stride=1, padding=0, bias=False)
    def forward(self, x):
        y = self.conv(x)
        y = self.batch_norm(y)
        y = self.relu(y)
        y = self.conv2(y)
        return y

class TinyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IdentityConv()
        # self.mu_layer = IdentityConv()
        # self.logvar_layer = IdentityConv()
        self.decoder = IdentityDeconv()
        self.sigmoid = nn.Sigmoid()

    """
    Notes:
    - architecture, kl weight, batch size, learning rate
    - torch interpolate function (maybe), transpose3d
    - convtranspose3d adds artifacts -> solve w conv3d layer after
    - activation fxn(s)
    - score based methods
    https://research.nvidia.com/labs/toronto-ai/CLD-SGM/
    - **denoiser
    - iddpm
    """

    def encode(self, x):
        # Take x and return the latent.
        x = self.encoder(x)
        return x
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        latent = self._reparametrize(mu, logvar)
        return latent, mu, logvar

    def decode(self, x):
        # Take latent and return the reconstructed x.
        x = self.decoder(x)
        x = self.sigmoid(x)
        return x

    def forward(self, x):
        # latent, mu, logvar = self.encode(x)
        latent = self.encode(x)
        x = self.decode(latent)
        return x, 0, 0

    # Private.

    def _reparametrize(self, mu, logvar):
        # Reparameterization trick: take random normal sample and scale it by the standard deviation, then shift it by the mean.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

if __name__ == "__main__":
    model = TinyAutoencoder()
    print(model)

    device = "cuda:1"
    model.to(device)

    # x = torch.randn(1, 3, 8, 64, 64)
    x = torch.randn(1, 3, 16, 128, 128)
    y, _, _ = model(x.to(device))
    assert x.shape == y.shape, f"{x.shape} != {y.shape}"
    print("Success!")

