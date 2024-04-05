import torch
from torchvision.models import resnet18

class TrivialAutoencoder(torch.nn.Module):
    """
    A trivial convolutional autoencoder.
    """
    def __init__(self):
        super(TrivialAutoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2),
            torch.nn.Sigmoid()  # output = [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    model = TrivialAutoencoder()
    print(model)

    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    assert x.shape == y.shape, f"{x.shape} != {y.shape}"
    print("Success!")
