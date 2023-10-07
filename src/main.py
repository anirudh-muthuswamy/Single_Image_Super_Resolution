import argparse
import ast
from preprocess import Preprocess

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

if __name__ == "__main__":
    main()

