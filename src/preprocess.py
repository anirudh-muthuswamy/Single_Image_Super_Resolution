''' Class that contains methods for creating high_res/low_res patches and preprocessing the dataset

CS7180 Advanced Perception     09/20/2023             Anirudh Muthuswamy, Gugan Kathiresan'''

from PIL import Image
import patchify
import pandas as pd
import glob as glob
import os
import cv2
import numpy as np


class Preprocess():

    def __init__(self,input_path):
        image_extension = 'png'
        self.image_paths = []
        self.image_paths.extend(glob.glob(os.path.join(input_path, f'*.{image_extension}')))

        ''' Function to store the high resolution patches, i.e, sub-images in the specified folder.
    High resolution images are simple patches of the original image of size 32x32 moved over the
    original image with a stride of 14.'''

    def create_high_res_patches(self, patch_path, patch_size, stride, num_images):

        os.makedirs(patch_path, exist_ok = True)

        print("Number of images in directory:", len(self.image_paths))

        breaktime = 0
        for image_path in self.image_paths:
            breaktime += 1
            ''' Obtaining the image'''
            input_image = Image.open(image_path)
            image_name = image_path.split(os.path.sep)[-1].split('.')[0]
            print(" Image name:", image_name)
            print(input_image.size)

            ''' Passing to patchify function'''
            high_res_patches = patchify.patchify(np.array(input_image), patch_size, stride)
            print(" \tImage size:", high_res_patches.shape)

            '''Reading the patches generated and saving individually'''
            counter = 0
            for i in range(high_res_patches.shape[0]):
                for j in range(high_res_patches.shape[1]):
                    counter +=1
                    patch = high_res_patches[i, j, 0, :, :, :]
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(f"{patch_path}/{image_name}_{counter}.png",patch)

            ''' "breaktime" Used to control the number of images in our dataset.
            To use the full dataset set the breaktime to the number of images in the dataset.
            For testing purposes, we chose to pick only 5 images, hence images = 5.'''

            if breaktime == num_images:
                break

    ''' Function to store the low resolution patches, i.e, sub-images that are
    generated from the high res patches.
    To acheive this we apply bilinear interpolation to high res patches and
    store the low res patches in a different directory.'''

    def create_low_res_patches(self, patch_path, patch_size, stride, num_images):

        os.makedirs(patch_path, exist_ok = True)
        breaktime = 0

        for image_path in self.image_paths:
            breaktime += 1
            ''' Obtaining the image'''
            input_image = Image.open(image_path)
            image_name = image_path.split(os.path.sep)[-1].split('.')[0]
            print(" Image name:", image_name)

            ''' Passing to patchify function'''
            high_res_patches = patchify.patchify(np.array(input_image), patch_size, stride)
            print(" \tImage size:", high_res_patches.shape)

            '''Reading the patches generated, applying Guassian Blur to reduce the quality,
                downsizing by half, resizing it back to the same size using bicubic interpolation
                and finally saving individually'''
            counter = 0
            for i in range(high_res_patches.shape[0]):
                for j in range(high_res_patches.shape[1]):
                    counter = counter + 1
                    patch = high_res_patches[i,j,0,:,:,:]
                    patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

                    h = 32
                    w = 32

                    # patch_blurred = cv2.GaussianBlur(patch,(3,3),0)
                    low_res_patch = cv2.resize(patch, (int(h*0.5), int(w*0.5)), interpolation=cv2.INTER_CUBIC)
                    upscaled_patch = cv2.resize(low_res_patch, (h,w), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(f"{patch_path}/{image_name}_{counter}.png",upscaled_patch)

            if breaktime == num_images:
                break

        ''' Function to store the low resolution image for the validation set, i.e, low res images
    generated from the original high res full image
    To acheive this we apply bilinear interpolation to size-halved high res image and
    store the low res image in a different directory.'''

    def create_low_res_images(self, low_res_image_path, images):

        os.makedirs(low_res_image_path, exist_ok = True)
        breaktime = 0

        for image_path in self.image_paths:
            breaktime += 1
            ''' Obtaining the image'''
            print(image_path)
            input_image = cv2.imread(image_path)
            image_name = image_path.split(os.path.sep)[-1].split('.')[0]
            # print(" Image name:", image_name)
            h = input_image.shape[1]
            w = input_image.shape[0]

            # image_blurred = cv2.GaussianBlur(input_image,(3,3),0)
            low_res_image = cv2.resize(input_image, (int(h*0.5), int(w*0.5)), interpolation=cv2.INTER_CUBIC)
            upscaled_image = cv2.resize(low_res_image, (h,w), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f"{low_res_image_path}/{image_name}.png",upscaled_image)

            if breaktime == images:
                break

    ''' Function to get patched dataset details and generate a csv file for the train data.'''

    def get_pandas_df(self, hr_path, lr_path, train_csv_path, num_images=-1):

        if num_images == -1:
            print("Number of high res files =", len(os.listdir(hr_path)))
            print("Number of low res files =", len(os.listdir(lr_path)))
        else:
            print("Number of high res files =", len(os.listdir(hr_path)[0:num_images]))
            print("Number of low res files =", len(os.listdir(lr_path)))

        if num_images == -1:
            y_filenames = os.listdir(hr_path)[0:len(os.listdir(hr_path))]
        else:
            y_filenames = os.listdir(hr_path)[0:num_images]
        y_filenames = sorted(y_filenames)
        y_filenames = [hr_path + file for file in y_filenames]
        print(len(y_filenames))

        if num_images == -1:
            x_filenames = os.listdir(lr_path)[0:len(os.listdir(lr_path))]
        else:
            x_filenames = os.listdir(lr_path)[0:num_images]
        x_filenames = sorted(x_filenames)
        x_filenames = [lr_path + file for file in x_filenames]

        ''' Adding all x filepaths == low res files and y filepaths == high res files to a dataframe.'''
        data = pd.DataFrame({'x_filepath':x_filenames, 'y_filepath':y_filenames})
        data_randomized = data.sample(frac=1, random_state=42).reset_index(drop=True)

        data_randomized.to_csv(train_csv_path, index = False)
