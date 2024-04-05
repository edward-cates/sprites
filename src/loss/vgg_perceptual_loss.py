import torch
from torchvision.models import vgg16

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = vgg16(pretrained=True).features
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        output_vgg, target_vgg = self.vgg(output), self.vgg(target)
        return torch.nn.functional.mse_loss(output_vgg, target_vgg)
