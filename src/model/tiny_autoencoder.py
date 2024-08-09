# https://github.com/madebyollin/taesd/blob/main/taesd.py

import torch

class EncoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, compress: bool):
        super().__init__()
        s = 2 if compress else 1
        self.layers = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=(1, s, s), padding=2, bias=False),
            torch.nn.ReLU(),
        )
    def forward(self, x):
        return self.layers(x)

# [b, 3, 8, 128, 128].
# [b, 128, 8, 2, 2]. -> 4096

class DecoderBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, decompress: bool, include_relu: bool = True):
        super().__init__()
        k = 6 if decompress else 5
        s = 2 if decompress else 1
        layers = [
            torch.nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(5, k, k), stride=(1, s, s), padding=2, bias=False),
        ]
        if include_relu:
            layers.append(torch.nn.ReLU())
        self.layers = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = torch.nn.Sequential(
            EncoderBlock(in_channels=3, out_channels=32, compress=False),
            EncoderBlock(in_channels=32, out_channels=32, compress=False),
            EncoderBlock(in_channels=32, out_channels=32, compress=True),
            EncoderBlock(in_channels=32, out_channels=32, compress=True),
        )
        self.block2 = torch.nn.Sequential(
            EncoderBlock(in_channels=32, out_channels=64, compress=False),
            EncoderBlock(in_channels=64, out_channels=64, compress=False),
            EncoderBlock(in_channels=64, out_channels=64, compress=True),
            EncoderBlock(in_channels=64, out_channels=64, compress=True),
        )
        self.block3 = torch.nn.Sequential(
            EncoderBlock(in_channels=64, out_channels=128, compress=False),
            EncoderBlock(in_channels=128, out_channels=128, compress=False),
            EncoderBlock(in_channels=128, out_channels=128, compress=True),
            EncoderBlock(in_channels=128, out_channels=128, compress=True),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block3 = torch.nn.Sequential(
            DecoderBlock(in_channels=128, out_channels=128, decompress=True),
            DecoderBlock(in_channels=128, out_channels=128, decompress=True),
            DecoderBlock(in_channels=128, out_channels=128, decompress=False),
            DecoderBlock(in_channels=128, out_channels=64, decompress=False),
        )
        self.block2 = torch.nn.Sequential(
            DecoderBlock(in_channels=64, out_channels=64, decompress=True),
            DecoderBlock(in_channels=64, out_channels=64, decompress=True),
            DecoderBlock(in_channels=64, out_channels=64, decompress=False),
            DecoderBlock(in_channels=64, out_channels=32, decompress=False),
        )
        self.block1 = torch.nn.Sequential(
            DecoderBlock(in_channels=32, out_channels=32, decompress=True),
            DecoderBlock(in_channels=32, out_channels=32, decompress=True),
            DecoderBlock(in_channels=32, out_channels=32, decompress=False),
            DecoderBlock(in_channels=32, out_channels=3, decompress=False, include_relu=False),
        )
    def forward(self, x):
        x = self.block3(x)
        x = self.block2(x)
        x = self.block1(x)
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
        # print("latent shape:", latent.shape)
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

