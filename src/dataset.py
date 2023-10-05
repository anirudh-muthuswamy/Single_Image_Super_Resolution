from SRCNNDataset import SRCNNDataset
from torch.utils.data import DataLoader, Dataset

''' CS7180 Advanced Perception     09/20/2023             Anirudh Muthuswamy, Gugan Kathiresan'''

class Dataset():
    def __init__(self):
        pass

    ''' Prepare the datasets by creating objects of the SRCNNDataset class that was previously defined'''
    def get_datasets(self, train_csv_file,valid_csv_file):

        dataset_train = SRCNNDataset(csv_file=train_csv_file)
        dataset_valid = SRCNNDataset(csv_file=valid_csv_file)

        return dataset_train, dataset_valid

    ''' Prepare the dataloader using the Dataloader torch library'''
    def get_dataloaders(self, dataset_train, dataset_valid):
        train_loader = DataLoader(
            dataset_train,
            batch_size=256,
            shuffle=True
        )
        valid_loader = DataLoader(
            dataset_valid,
            batch_size=1,
            shuffle=False
        )
        return train_loader, valid_loader