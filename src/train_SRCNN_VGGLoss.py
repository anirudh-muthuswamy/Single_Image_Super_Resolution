''' CS7180 Advanced Perception     09/20/2023             Anirudh Muthuswamy, Gugan Kathiresan'''
from PIL import Image
from tqdm import tqdm
import time
import patchify

import pandas as pd
import glob as glob
import os
import cv2
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import save_image
plt.style.use('ggplot')
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
from torchsummary import summary
from CustomConvLayer import CustomConvLayer
from utils import Utils

if __name__ == "__main__":
    epochs = 10
    lr = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    utils = Utils()

    SAVE_VALIDATION_RESULTS = True

    ''' Initialising the VGG model and printing its layers. This gives us information
    on which layer should we use for the perceptual loss.'''

    vgg = models.vgg19(pretrained=True).features.eval()
    vgg.to(device = 'cuda')

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
    for param in vgg.parameters():
        param.requires_grad = False

    class VGGPerceptualLoss(nn.Module):
        def __init__(self):
            super(VGGPerceptualLoss, self).__init__()
            self.vgg = vgg

        def forward(self, x, y):
            loss = 0
            x_features = self.vgg[35](x)
            y_features = self.vgg[35](y)
            # Calculate Euclidean distance:
            loss += torch.norm(x_features - y_features)

            return loss



    model = CustomConvLayer().to(device)
    summary(model,(3,32,32))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = VGGPerceptualLoss()

    train_csv_file = '/content/train_data_5_Imgs_patched.csv'
    valid_csv_file = '/content/test_data_50_Imgs.csv'
    output_dir = 'output_SRCNN_VGG_loss'

    utils.start_training(model, epochs, optimizer, criterion, train_csv_file, valid_csv_file, output_dir)