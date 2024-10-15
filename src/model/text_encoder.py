import torch

class TextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torch.nn.Embedding(
            num_embeddings=5,
            embedding_dim=128*30*2*2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        # reshape to [b, 128, 30, 2, 2]
        return x.view(x.size(0), 128, 30, 2, 2)
