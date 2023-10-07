''' CS7180 Advanced Perception     09/20/2023             Anirudh Muthuswamy, Gugan Kathiresan'''

import glob as glob
import matplotlib.pyplot as plt

import torch
plt.style.use('ggplot')

import torch.optim as optim
from torchsummary import summary
from models.SRCNN import SRCNN
from utils import Utils
from models.VGG_PERCEPTUAL import VGGPerceptualLoss


if __name__ == "__main__":
    epochs = 10
    lr = 0.001
    device = 'mps' if torch.cuda.is_available() else 'cpu'
    utils = Utils()

    SAVE_VALIDATION_RESULTS = True

    model = SRCNN().to(device)
    summary(model,(3,32,32))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = VGGPerceptualLoss()

    train_csv_file = '/content/train_data_5_Imgs_patched.csv'
    valid_csv_file = '/content/test_data_50_Imgs.csv'
    output_dir = 'output_SRCNN_VGG_loss'

    utils.start_training(model, epochs, optimizer, criterion, train_csv_file, valid_csv_file, output_dir)