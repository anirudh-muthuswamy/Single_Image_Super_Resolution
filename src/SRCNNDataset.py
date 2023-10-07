'''Class that defines a custom dataset that reads the Image files from a pandas dataframe.

CS7180 Advanced Perception     09/20/2023             Anirudh Muthuswamy, Gugan Kathiresan'''


import torch
from PIL import Image

import pandas as pd
import glob as glob
import numpy as np

from torch.utils.data import Dataset


device = torch.device('mps')

'''Class to perform the necessary initial preprocessing and to call a dataset from a csv file (standard pytorch dataset).
 Performs normalization and creates a dataset tensor for images and labels separately'''

class SRCNNDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform


    def __len__(self):
        return len(self.data)

    ''' returns a tensor for the images and labels of the datasets (low res and high res iamges respectively)
     Performs normalization and transpose that brings it to the requirements of a tensor.'''
    def __getitem__(self, idx):
        image = Image.open(self.data.iloc[idx, 0]).convert('RGB')
        label = Image.open(self.data.iloc[idx, 1]).convert('RGB')

        image = np.array(image, dtype=np.float32)
        label = np.array(label, dtype=np.float32)

        image /= 255.
        label /= 255.

        image = image.transpose([2, 0, 1])
        label = label.transpose([2, 0, 1])

        return (
            torch.tensor(image, dtype=torch.float),
            torch.tensor(label, dtype=torch.float)
        )