
import math
from tqdm import tqdm
import time

import glob as glob
import os
import math

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image
plt.style.use('ggplot')
from torch.utils.data import Dataset

from dataset import Dataset

device = torch.device('cuda')

class Utils():
    def __init__(self):
        os.makedirs('outputs', exist_ok = True)
        self.data = Dataset()

    ''' Implementing the PSNR Metric'''

    def psnr(self, label, outputs, max_val = 1):
        # Formula for psnr = 20 * log(Max pixel value) - 10 * log(MSE)

        label = label.cpu().detach().numpy()
        outputs = outputs.cpu().detach().numpy()

        diff = outputs - label
        rmse = math.sqrt(np.mean((diff)**2))
        if rmse == 0:
            return 100
        else:
            return 20 * math.log10(max_val / rmse)

    '''Function to save graphs that track the training and validaton loss and psnr'''


    def save_plot(self, train_loss, val_loss, train_psnr, val_psnr, output_dir):
        os.makedirs(output_dir, exist_ok = True)
        # Loss plots.
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color='orange', label='train loss')
        plt.plot(val_loss, color='red', label='validataion loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{output_dir}/loss.png')
        plt.close()
        # PSNR plots.
        plt.figure(figsize=(10, 7))
        plt.plot(train_psnr, color='green', label='train PSNR dB')
        plt.plot(val_psnr, color='blue', label='validataion PSNR dB')
        plt.xlabel('Epochs')
        plt.ylabel('PSNR (dB)')
        plt.legend()
        plt.savefig(f'{output_dir}/psnr.png')
        plt.close()

    '''Function to save a model to a torch pth file'''

    def save_model_state(self, model, output_dir):
        # save the model to disk
        os.makedirs(output_dir, exist_ok = True)
        print('Saving model...')
        torch.save(model.state_dict(), f'{output_dir}/model.pth')

    '''Function to save a model checkpoint and remove any previous model checkpoints
    if existing.Useful to pickup and continue training'''

    def save_model(self, epochs, model, optimizer, criterion, output_dir):
        os.makedirs(output_dir, exist_ok = True)
        torch.save({
                    'epoch': epochs+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion,
                    }, f"{output_dir}/model_ckpt.pth")


    '''Function to save the validation reconstructed images.'''

    def save_validation_results(self, outputs, epoch, batch_iter, output_dir):
        os.makedirs(output_dir+'/valid_results', exist_ok=True)
        save_image(
            outputs,
            f"{output_dir}/valid_results/val_sr_{epoch}_{batch_iter}.png"
        )

    def train(self, model, dataloader, optimizer, criterion):

        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_data = data[0].to(device)
            label = data[0].to(device)

            optimizer.zero_grad()
            outputs = model(image_data)
            loss = criterion(outputs, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batch_psnr = self.psnr(label, outputs)
            running_psnr += batch_psnr

        final_loss = running_loss/len(dataloader.dataset)
        final_psnr = running_psnr/len(dataloader)
        return final_loss, final_psnr

    def validate(self, model, dataloader, epoch, criterion, output_dir):
        SAVE_VALIDATION_RESULTS = True
        model.eval()
        running_loss = 0.0
        running_psnr = 0.0

        with torch.no_grad():
            for bi, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                image_data = data[0].to(device)
                label = data[1].to(device)

                outputs = model(image_data)
                loss = criterion(outputs, label)

                running_loss += loss.item()
                batch_psnr = self.psnr(label, outputs)
                running_psnr += batch_psnr

                # For saving the batch samples for the validation results
                # every epoch.
                if SAVE_VALIDATION_RESULTS and (epoch%10==0):
                    self.save_validation_results(outputs, epoch, bi, output_dir)

            final_loss = running_loss/len(dataloader.dataset)
            final_psnr = running_psnr/len(dataloader)
            return final_loss, final_psnr

    def start_training(self, model, epochs, optimizer, criterion, train_csv_file, valid_csv_file, output_dir):

        dataset_train, dataset_valid = self.data.get_datasets(train_csv_file,valid_csv_file)

        train_loader, valid_loader = self.data.get_dataloaders(dataset_train, dataset_valid)
        print(f"Training samples: {len(dataset_train)}")
        print(f"Validation samples: {len(dataset_valid)}")

        train_loss, val_loss = [], []
        train_psnr, val_psnr = [], []
        start = time.time()
        for epoch in range(epochs):
            train_epoch_loss, train_epoch_psnr = self.train(model, train_loader, optimizer, criterion)
            val_epoch_loss, val_epoch_psnr = self.validate(model, valid_loader, epoch+1, criterion, output_dir)
            print(f"Train PSNR: {train_epoch_psnr:.3f}")
            print(f"Val PSNR: {val_epoch_psnr:.3f}")

            train_loss.append(train_epoch_loss)
            train_psnr.append(train_epoch_psnr)
            val_loss.append(val_epoch_loss)
            val_psnr.append(val_epoch_psnr)

            # Save model with all information every 10 epochs. Can be used
            # resuming training.
            if (epoch+1) % 10 == 0:
                self.save_model(epoch, model, optimizer, criterion, output_dir)
            # Save the model state dictionary only every epoch. Small size,
            # can be used for inference.
            self.save_model_state(model, output_dir)
            # Save the PSNR and loss plots every epoch.
            self.save_plot(train_loss, val_loss, train_psnr, val_psnr, output_dir)

        end = time.time()
        print(f"Finished training in: {((end-start)/60):.3f} minutes")