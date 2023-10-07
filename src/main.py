import argparse
import ast
from preprocess import Preprocess
import torch
import torch.optim as optim
import torch.nn as nn
from utils import Utils
from models.SRCNN import SRCNN
from models.SRCNN_VAR_FILTERS import SRCNN_VAR_FILTERS
from models.VGG_PERCEPTUAL import VGGPerceptualLoss

def import_class(class_name):
    module = __import__(class_name.lower())
    return getattr(module, class_name)

def main():
    parser = argparse.ArgumentParser(description="Description of your command-line tool")
    
    parser.add_argument("--task", help="Calls the class defined in the src folder")


    parser.add_argument("--input_path", help="With the input path")
    parser.add_argument("--stride", help="Stride for patch creation")
    parser.add_argument("--patch_size", help="Enter patch size (h, w, c)")
    parser.add_argument("--num_images", help='num of images to be patched')
    parser.add_argument("--method", help='calls preprocess method')
    parser.add_argument("--low_res_path", help = 'path to low res files')
    parser.add_argument("--high_res_path", help = 'path to high res files')
    parser.add_argument("--output_csv_path", help = 'path to output CSV files')

    parser.add_argument("--model", help="enter the model class")
    parser.add_argument("--lr", help = 'enter the learning rate')
    parser.add_argument("--loss",help='enter the loss function to use')
    parser.add_argument("--epochs", help = 'enter the number of epochs to run the model')
    parser.add_argument("--train_csv_file", help = 'path to training CSV file')
    parser.add_argument("--valid_csv_file", help='path to testing CSV file')
    parser.add_argument('--output_dir',help='enter output director to store results')
    
    args = parser.parse_args()

    if args.task=='preprocess' and args.input_path:
        preprocess = Preprocess(input_path=args.input_path)

        if args.method == 'create_high_res_patches':
            if args.stride and args.patch_size and args.num_images:
                patch_size_tuple = ast.literal_eval(args.patch_size)
                preprocess.create_high_res_patches(f"high_res_patches_{args.num_images}_imgs", 
                                                   patch_size_tuple, 
                                                   int(args.stride), 
                                                   int(args.num_images))
            else: 
                print('inavlid arguments')
                
        elif args.method == 'create_low_res_patches':
            if args.stride and args.patch_size and args.num_images:
                patch_size_tuple = ast.literal_eval(args.patch_size)
                preprocess.create_low_res_patches(f"low_res_patches_{args.num_images}_imgs", 
                                                   patch_size_tuple, 
                                                   int(args.stride), 
                                                   int(args.num_images))
            else: 
                print('inavlid arguments')
        
        elif args.method == 'create_low_res_images':
            if args.num_images:
                preprocess.create_low_res_images(f"low_res_{args.num_images}_imgs", 
                                                 int(args.num_images))
            else: 
                print('inavlid arguments')
        
        elif args.method == 'get_pandas_df':
            if args.low_res_path and args.high_res_path and args.output_csv_path:
                if args.num_images:
                    preprocess.get_pandas_df(args.high_res_path, 
                                         args.low_res_path, 
                                         args.output_csv_path, 
                                         int(args.num_images))
                else:
                    preprocess.get_pandas_df(args.high_res_path, 
                                         args.low_res_path, 
                                         args.output_csv_path)
            else: 
                print('inavlid arguments')
                    
        else:
            print('invalid method')
    
    elif args.task == 'train':
        if args.model and args.lr and args.loss and args.epochs \
            and args.train_csv_file and args.valid_csv_file and args.output_dir:

            device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            
            if args.loss == 'mse':
                criterion = nn.MSELoss()
            elif args.loss =='vgg':
                criterion = VGGPerceptualLoss()

            utils = Utils()

            if args.model == 'SRCNN':
                model = SRCNN().to(device)
            elif args.model == 'SRCNN_VAR_FILTERS':
                model = SRCNN_VAR_FILTERS.to(device)

            optimizer = optim.Adam(model.parameters(), float(args.lr))

            utils.start_training(model,
                                 int(args.epochs),
                                 optimizer,
                                 criterion,
                                 args.train_csv_file,
                                 args.valid_csv_file,
                                 args.output_dir)
        else:
            print('invalid arguments')
    else:
        print('invalid task')
            
if __name__ == "__main__":
    main()

    #to replicate:

    #python src/main.py --task preprocess 
    #                   --input_path src/DIV2K2017/DIV2K/DIV2K_train_HR 
    #                   --method create_high_res_patches 
    #                   --stride 14 
    #                   --patch_size "(32,32,3)"  
    #                   --num_images 5

    #python src/main.py --task preprocess 
    #                   --input_path src/DIV2K2017/DIV2K/DIV2K_train_HR 
    #                   --method create_low_res_patches  
    #                   --stride 14 
    #                   --patch_size "(32,32,3)"  
    #                   --num_images 5

    #python src/main.py --task preprocess 
    #                   --input_path src/DIV2K2017/DIV2K/DIV2K_train_HR 
    #                   --method create_low_res_images 
    #                   --num_images 50

    #python src/main.py --task preprocess 
    #                   --input_path src/DIV2K2017/DIV2K/DIV2K_train_HR 
    #                   --method get_pandas_df 
    #                   --high_res_path high_res_patches_5_imgs 
    #                   --low_res_path low_res_patches_5_imgs 
    #                   --output_csv_path train_data_5_imgs_patched.csv

    # python src/main.py --task preprocess 
    #                   --input_path src/DIV2K2017/DIV2K/DIV2K_train_HR 
    #                   --method get_pandas_df 
    #                   --high_res_path src/DIV2K2017/DIV2K/DIV2K_train_HR  
    #                   --low_res_path low_res_50_imgs  
    #                   --output_csv_path test_data_50_imgs.csv 
    #                   --num_images 50

    #Training SRCNN with MSE

    #python src/main.py --task train 
    #                   --model SRCNN 
    #                   --lr 0.001 
    #                   --epochs 10 
    #                   --loss mse 
    #                   --train_csv_file train_data_5_imgs_patched.csv 
    #                   --valid_csv_file test_data_50_imgs.csv 
    #                   --output_dir outputs/SRCNN_MSE_LOSS    

    #Training SRCNN_VAR_FILTERS with MSE

    #python src/main.py --task train 
    #                   --model SRCNN_VAR_FILTERS
    #                   --lr 0.001 
    #                   --epochs 10 
    #                   --loss mse 
    #                   --train_csv_file train_data_5_imgs_patched.csv 
    #                   --valid_csv_file test_data_50_imgs.csv 
    #                   --output_dir outputs/SRCNN_VAR_FILTERS_MSE_LOSS    

    #Training SRCNN_VAR_FILTERS with VGG Perceptual Loss

    #python src/main.py --task train 
    #                   --model SRCNN_VAR_FILTERS
    #                   --lr 0.001 
    #                   --epochs 10 
    #                   --loss vgg 
    #                   --train_csv_file train_data_5_imgs_patched.csv 
    #                   --valid_csv_file test_data_50_imgs.csv 
    #                   --output_dir outputs/SRCNN_VAR_FILTERS_VGG_LOSS 




    




