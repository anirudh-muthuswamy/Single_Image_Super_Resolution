''' SRCNN Model v2.
 Includes some modifications made by Anirudh + Gugan.
 The ideology behind the approach originates from the basic task of creating a wider but not-heavy network.
 There are four parallel initial convolutional layers that vary in kernel size. These extract different perspectives
 of different patches from the upsampled low res input (similar to the base SRCNN but with different filter sizes).

 These patches represent different features maps highlight unique information from the input image for further operations.
 By applying a wider network with parallel layers, which are later concated to perform non-linear mapping,
 our goal is to pickup features that were possibly not picked up by previous single layer filter sizes.
  CS7180 Advanced Perception     09/20/2023             Anirudh Muthuswamy, Gugan Kathiresan
 '''

import torch
import torch.nn as nn
import torch.nn.functional as F

class SRCNN_VAR_FILTERS(nn.Module):
    def __init__(self):
        super(SRCNN_VAR_FILTERS, self).__init__()
        self.conv_input1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=9, padding=4)
        self.conv_input2 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv_input3 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=5, padding=2)
        self.conv_input4 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(
            in_channels = 256, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels = 32, out_channels=3, kernel_size=5, padding=2)
        
    def forward(self, x):
        out1 = F.relu(self.conv_input1(x))
        out2 = F.relu(self.conv_input2(x))
        out3 = F.relu(self.conv_input3(x))
        out4 = F.relu(self.conv_input4(x))
        x = torch.cat((out1, out2, out3, out4), 1)
        x = F.relu(self.conv2(x))
        
        return self.conv3(x)