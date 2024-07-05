# import matplotlib
# matplotlib.use('agg')
# import torch
# import argparse
# import numpy as np
# import importlib
# import torch.optim
# import torch.nn.functional as F
# import torch.nn as nn
# from torch.autograd import Variable
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# from torch.utils.data import random_split
# import hyperGAN_model
# import feature_model
# import random

# def load_args():

#     parser = argparse.ArgumentParser(description='HyperNet')
#     parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
#     parser.add_argument('--num_models', default=2, type=int, help='input images channel') 
#     parser.add_argument('--in_channels', default=3, type=int, help='input images channel')
#     parser.add_argument('--input_size', default=256, type=int, help='input images size')
#     parser.add_argument('--num_classes', default=20, type=int, help='number of classes')
#     parser.add_argument('--z', default=64, type=int, help='Q(z|s) latent space width')
#     parser.add_argument('--s_mean', default=0, type=int, help='S sample mean')
#     parser.add_argument('--s_std', default=1, type=int, help='S sample standard deviation')
#     parser.add_argument('--s', default=256, type=int, help='S sample dimension')
#     parser.add_argument('--bias', action='store_true', help='Include HyperNet bias')
#     parser.add_argument('--batch_size', default=16, type=int, help='network batch size')
#     parser.add_argument('--epochs', default=200000, type=int)
#     parser.add_argument('--ngen', default=4, type=int)
#     parser.add_argument('--resume', default=None, type=str, help='resume from path')
#     parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
#     parser.add_argument('--wd', default=5e-4, type=float, help='weight decay (optimizer)')
#     parser.add_argument('--cuda', action='store_true')
#     parser.add_argument('--diversity_lambda',type=int, default=1, help='diversity lambda')
 
#     args = parser.parse_args()
#     return args

# def generate(args):  

#     '''set random seed and device'''
#     if args.manualSeed is None:
#         args.manualSeed = random.randint(1, 10000)
#     print("Random Seed: ", args.manualSeed)
#     random.seed(args.manualSeed)
#     np.random.seed(args.manualSeed)        
#     if args.cuda and torch.cuda.is_available():
#         torch.cuda.manual_seed_all(args.manualSeed)
#         args.device = 'cuda:0'
#         print("cuda")
#     else:
#         torch.manual_seed(args.manualSeed)
#         args.device = 'cpu'
#         print("cpu")


#     print(args.device)

#     hypergan =  hyperGAN_model.HyperGAN(args)
#     generator = hypergan.generator
#     mixer = hypergan.mixer
#     hypergan.restore_models(args)

#     feature = feature_model.FaceRecognitionModel(num_classes=20)


#     for i in range(args.num_models):
#         with torch.no_grad():       
#             s = torch.normal(mean=args.s_mean, std=args.s_std, size=(2, args.s)).to(args.device)
#             # print(s)
#             codes = mixer(s)
#             codes = codes.view(2, args.ngen, args.z)
#             codes = torch.stack([codes[:, i] for i in range(args.ngen)])
#             params = generator(codes)

#             for new_model_param, loaded_param in zip(feature.parameters(), params):
#                 new_model_param.copy_(loaded_param)
#             torch.save(feature.state_dict(), 'saved_models/generated_model{}.pth'.format(i))
        
# if __name__ == '__main__':
#     args = load_args()
#     generate(args)

import matplotlib
matplotlib.use('agg')
import torch
import argparse
import numpy as np
import importlib
import torch.optim
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
import feature_model_med_1
import random
import os
import re

def load_args():

    parser = argparse.ArgumentParser(description='HyperNet')
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
    parser.add_argument('--num_models', default=16, type=int, help='number of modles') 
    parser.add_argument('--in_channels', default=3, type=int, help='input images channel')
    parser.add_argument('--input_size', default=256, type=int, help='input images size')
    parser.add_argument('--num_classes', default=20, type=int, help='number of classes')
    parser.add_argument('--z', default=64, type=int, help='Q(z|s) latent space width')
    parser.add_argument('--s_mean', default=0, type=int, help='S sample mean')
    parser.add_argument('--s_std', default=1, type=int, help='S sample standard deviation')
    parser.add_argument('--s', default=256, type=int, help='S sample dimension')
    parser.add_argument('--bias', action='store_true', help='Include HyperNet bias')
    parser.add_argument('--batch_size', default=32, type=int, help='network batch size')
    parser.add_argument('--epochs', default=200000, type=int)
    parser.add_argument('--ngen', default=4, type=int)
    parser.add_argument('--resume', default=None, type=str, help='resume from path')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay (optimizer)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--diversity_lambda',type=int, default=1, help='diversity lambda')
    parser.add_argument('--threshold',type=float, default=0.1, help='diversity threshold')
    parser.add_argument('--root_dir',type=str, default=None, help='root directory')
 
    args = parser.parse_args()
    return args

def generate(args):  
    '''set random seed and device'''
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)     
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)        
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
        args.device = 'cuda:0'
        print("cuda")
    else:
        torch.manual_seed(args.manualSeed)
        args.device = 'cpu'
        print("cpu")


    for m_idx, masknets in enumerate(os.listdir(args.root_dir)):
        masknets_path = os.path.join(args.root_dir, masknets)
        mask_ls = masknets.split('_')
        args.epochs = int(mask_ls[1])
        args.diversity_lambda = int(mask_ls[3])
        match = re.match(r'([\d.]+)\.pth', mask_ls[5])
        args.threshold = float(match.group(1))
            
        hypergan =  feature_model_med_1.HyperGAN(args)
        generator = hypergan.generator
        mixer = hypergan.mixer
        fc = hypergan.FC

        hypergan.restore_models(args)
        feature = feature_model_med_1.FaceRecognitionModelFeatureExtractor(num_classes=20)
        classifier = feature_model_med_1.FaceRecognitionModel(num_classes=20)
        
        with torch.no_grad():
            res = 0.0       
            for _ in range(10):
            # s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.num_models, args.s)).to(args.device)
                s = torch.empty(args.num_models, args.s).normal_(mean=args.s_mean, std=args.s_std)
                # print(s)
                codes = mixer(s)
                codes = codes.view(args.num_models, args.ngen, args.z)
                codes = torch.stack([codes[:, i] for i in range(args.ngen)])
                params = generator(codes)
                var = torch.var(torch.flatten(params[0],start_dim=1,end_dim=-1),dim=0).mean()+torch.var(torch.flatten(params[2],start_dim=1,end_dim=-1),dim=0).mean()+torch.var(torch.flatten(params[4],start_dim=1,end_dim=-1),dim=0).mean()
                print(f"The variance of the parameters epoch_{args.epochs}_div_{args.diversity_lambda}_thre_{args.threshold} is {var}")
                res += var

            res /= 10.0
            with open(f"output/mse_batch.txt", 'a') as f:      
                f.write(f"\nThe variance of the parameters epoch_{args.epochs}_div_{args.diversity_lambda}_thre_{args.threshold} is {res}")
            params = list(zip(*params))
            
            
            for i in range(args.num_models):
                for new_model_param, loaded_param in zip(feature.parameters(), params[i]):
                    new_model_param.copy_(loaded_param)
                for new_param, loaded_param in zip(classifier.parameters(), feature.parameters()):
                    if new_param.data.shape == loaded_param.data.shape:
                        new_param.data.copy_(loaded_param.data)
                for new_param, loaded_param in zip(classifier.fc2.parameters(), fc.parameters()):
                    new_param.data.copy_(loaded_param.data)
                
                save_dir = f"MaskNet/hypernet/model_1/sensitivities"
                load_path = os.path.join(save_dir, f"epoch_{args.epochs}_div_{args.diversity_lambda}_thre_{args.threshold}/generated_model{i}.pth")
                save_dir = os.path.dirname(load_path)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(classifier.state_dict(), load_path)

    
if __name__ == '__main__':
    args = load_args()
    generate(args)