# https://github.com/madebyollin/taesd/blob/main/taesd.py

import torch

class TinyAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv3d(3, 64, kernel_size=5, stride=2, padding=2, bias=False),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),
        )

        # self.mu_layer = IdentityConv()
        # self.logvar_layer = IdentityConv()

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 3, kernel_size=6, stride=2, padding=2, bias=False),
            torch.nn.BatchNorm3d(3),
            torch.nn.ReLU(),
        )

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
        # clamp to 0-1.
        x = torch.clamp(x, 0, 1)
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

