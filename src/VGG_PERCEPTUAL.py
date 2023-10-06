''' Load pre-trained VGG-19 model without fully connected layers

Includes some modifications made by Anirudh + Gugan. Swaps MSE loss for
the VGG Perceptual Loss inspired from the SRGAN paper.
---
Christian Ledig, Lucas Theis, Ferenc Husza ́r, Jose Caballero, Andrew
Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz,
Zehan Wang,Wenzhe Shi, "Photo-Realistic Single Image Super-Resolution Using a
Generative Adversarial Network Actions", CVPR, 2017.
---
'''

import torch
import torch.nn as nn
import torchvision.models as models


class VGGPerceptualLoss(nn.Module):
    def __init__(self):

        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.eval()
        vgg.to(device = 'cuda')

        for param in vgg.parameters():
            param.requires_grad = False

        self.vgg = vgg

    def forward(self, x, y):
        loss = 0
        x_features = self.vgg[35](x)
        y_features = self.vgg[35](y)
        # Calculate Euclidean distance:
        loss += torch.norm(x_features - y_features)

        return loss
