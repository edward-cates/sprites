# https://github.com/madebyollin/taesd/blob/main/taesd.py

import torch

class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=(1, 2, 2), padding=2, bias=False),
            torch.nn.ReLU(),
        )
    def forward(self, x):
        return self.layers(x)

# [b, 3, 8, 128, 128].
# [b, 32, 8, 32, 32].

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, include_relu: bool):
        super().__init__()
        layers = [
            torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(5, 6, 6), stride=(1, 2, 2), padding=2, bias=False),
        ]
        if include_relu:
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = EncoderBlock(in_channels=3, out_channels=32)
        self.conv2 = EncoderBlock(in_channels=32, out_channels=32)
        self.conv3 = EncoderBlock(in_channels=32, out_channels=64)
        self.conv4 = EncoderBlock(in_channels=64, out_channels=64)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv4 = DecoderBlock(in_channels=64, out_channels=64, include_relu=True)
        self.conv3 = DecoderBlock(in_channels=64, out_channels=32, include_relu=True)
        self.conv2 = DecoderBlock(in_channels=32, out_channels=32, include_relu=True)
        self.conv1 = DecoderBlock(in_channels=32, out_channels=3, include_relu=False)
    def forward(self, x):
        x = self.conv2(x)
        x = self.conv1(x)
        # clamp to 0-1.
        x = torch.clamp(x, 0, 1)
        return x

class TinyAutoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        # self.mu_layer = IdentityConv()
        # self.logvar_layer = IdentityConv()

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
        return x

    def forward(self, x):
        # latent, mu, logvar = self.encode(x)
        latent = self.encode(x)
        y = self.decode(latent)
        assert x.shape == y.shape, f"{x.shape} != {y.shape}"
        return y, 0, 0

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

    x = torch.randn(1, 3, 8, 128, 128)
    y, _, _ = model(x.to(device))
    assert x.shape == y.shape, f"{x.shape} != {y.shape}"
    print("Success!")

