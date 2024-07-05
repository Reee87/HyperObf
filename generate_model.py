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
import time
import os

def load_args():

    parser = argparse.ArgumentParser(description='HyperNet')
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
    parser.add_argument('--num_models', default=16, type=int, help='input images channel') 
    parser.add_argument('--in_channels', default=3, type=int, help='input images channel')
    parser.add_argument('--input_size', default=256, type=int, help='input images size')
    parser.add_argument('--num_classes', default=20, type=int, help='number of classes')
    parser.add_argument('--z', default=64, type=int, help='Q(z|s) latent space width')
    parser.add_argument('--s_mean', default=0, type=int, help='S sample mean')
    parser.add_argument('--s_std', default=1, type=int, help='S sample standard deviation')
    parser.add_argument('--s', default=256, type=int, help='S sample dimension')
    parser.add_argument('--bias', action='store_true', help='Include HyperNet bias')
    parser.add_argument('--batch_size', default=32, type=int, help='network batch size')
    parser.add_argument('--epochs', default=850, type=int)
    parser.add_argument('--ngen', default=4, type=int)
    parser.add_argument('--resume', default=None, type=str, help='resume from path')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='weight decay (optimizer)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--diversity_lambda',type=int, default=1, help='diversity lambda')
    parser.add_argument('--threshold',type=float, default=0.01, help='diversity threshold')
 
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

    print(args.device)

    hypergan =  feature_model_med_1.HyperGAN(args)
    generator = hypergan.generator
    mixer = hypergan.mixer.to(args.device)
    fc = hypergan.FC.to(args.device)

    hypergan.restore_models(args)
    feature = feature_model_med_1.FaceRecognitionModelFeatureExtractor(num_classes=20)
    classifier = feature_model_med_1.FaceRecognitionModel(num_classes=20)

    # record the start time
    start_time = time.time()


    with torch.no_grad():       
        # s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.num_models, args.s)).to(args.device)
        s = torch.empty(args.num_models, args.s).normal_(mean=args.s_mean, std=args.s_std).to(args.device)
        # print(s)
        codes = mixer(s)
        codes = codes.view(args.num_models, args.ngen, args.z)
        codes = torch.stack([codes[:, i] for i in range(args.ngen)])
        params = generator(codes)
        # var = torch.var(torch.flatten(params[0],start_dim=1,end_dim=-1),dim=0).mean()+torch.var(torch.flatten(params[2],start_dim=1,end_dim=-1),dim=0).mean()+torch.var(torch.flatten(params[4],start_dim=1,end_dim=-1),dim=0).mean()

        params = list(zip(*params))
        
        
        for i in range(args.num_models):
            for new_model_param, loaded_param in zip(feature.parameters(), params[i]):
                new_model_param.copy_(loaded_param)
            for new_param, loaded_param in zip(classifier.parameters(), feature.parameters()):
                if new_param.data.shape == loaded_param.data.shape:
                    new_param.data.copy_(loaded_param.data)
            for new_param, loaded_param in zip(classifier.fc2.parameters(), fc.parameters()):
                new_param.data.copy_(loaded_param.data)
            
            # save_dir = f"MaskNet/hypernet/model_1/sensitivities"
            # load_path = os.path.join(save_dir, f"epoch_{args.epochs}_div_{args.diversity_lambda}_thre_{args.threshold}/generated_model{i}.pth")
            # save_dir = os.path.dirname(load_path)
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # torch.save(classifier.state_dict(), load_path)

    end_time = time.time()

    running_time = end_time - start_time
    print("The latency is :", running_time, "second")

    
if __name__ == '__main__':
    args = load_args()
    
    generate(args)