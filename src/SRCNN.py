'''Class to define the base SRCNN Model from the source paper inspiration.
 Uses 3 convolutional layers with filter kernel sizes and channels as suggested by the paper.

 C. Dong, C. C. Loy, K. He, and X. Tang, “Learning a Deep Convolutional Network
 for Image Super-Resolution,” Computer Vision – ECCV 2014, pp. 184–199, 2014,
 doi: https://doi.org/10.1007/978-3-319-10593-2_13.
 
 CS7180 Advanced Perception     09/20/2023             Anirudh Muthuswamy, Gugan Kathiresan
'''

import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(
            in_channels = 64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels = 32, out_channels=3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)