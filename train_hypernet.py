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

def load_args():

    parser = argparse.ArgumentParser(description='HyperNet')
    parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
    parser.add_argument('--in_channels', default=3, type=int, help='input images channel')
    parser.add_argument('--input_size', default=256, type=int, help='input images size')
    parser.add_argument('--num_classes', default=20, type=int, help='number of classes')
    parser.add_argument('--z', default=64, type=int, help='Q(z|s) latent space width')
    parser.add_argument('--s_mean', default=0, type=int, help='S sample mean')
    parser.add_argument('--s_std', default=1, type=int, help='S sample standard deviation')
    parser.add_argument('--s', default=256, type=int, help='S sample dimension')
    parser.add_argument('--bias', action='store_true', help='Include HyperNet bias')
    parser.add_argument('--batch_size', default=16, type=int, help='network batch size')
    parser.add_argument('--epochs', default=850, type=int)
    parser.add_argument('--ngen', default=4, type=int)
    parser.add_argument('--resume', default=None, type=str, help='resume from path')
    parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=5e-3, type=float, help='weight decay (optimizer)')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--diversity_lambda',type=int, default=1, help='diversity lambda')
    parser.add_argument('--threshold',type=float, default=0.1, help='diversity threshold')
   
    args = parser.parse_args()
 
    return args

def find_majority(k):
    myMap = {}
    maximum = ( '', 0 ) # (occurring element, occurrences)
    for n in k:
        if n in myMap: myMap[n] += 1
        else: myMap[n] = 1

        # Keep track of maximum on the go
        if myMap[n] > maximum[1]: maximum = (n,myMap[n])

    return maximum

def train(args):

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
    '''instantiate attack model and HyperNet'''
    hypergan =  feature_model_med_1.HyperGAN(args)
    generator = hypergan.generator
    mixer = hypergan.mixer
    fc = hypergan.FC
    if args.resume is not None:
        hypergan.restore_models(args)


    """ attach optimizers """
    # optimizerAttacker = torch.optim.Adam(netAttacker.parameters(), lr=args.attack_lr, betas=(args.attack_beta1, 0.999), weight_decay=args.attack_l2reg)
    
    optimQ = torch.optim.Adam(mixer.parameters(), lr=args.lr,weight_decay=args.wd)
    optimW = []
    # for m in range(args.ngen):
    #     s = getattr(generator, 'W{}'.format(m+1))
    #     optimW.append(torch.optim.Adam(s.parameters(), lr=args.lr, weight_decay=args.wd))
    s = getattr(generator, 'W1')
    optimW.append(torch.optim.Adam(s.parameters(), lr=args.lr, weight_decay=args.wd))
    s = getattr(generator, 'W4')
    optimW.append(torch.optim.Adam(s.parameters(), lr=args.lr, weight_decay=args.wd))
    s = getattr(generator, 'W5')
    optimW.append(torch.optim.Adam(s.parameters(), lr=args.lr, weight_decay=args.wd))
    # s = getattr(generator, 'W6')
    # optimW.append(torch.optim.Adam(s.parameters(), lr=args.lr, weight_decay=args.wd))

    optimF = torch.optim.Adam(fc.parameters(), lr=args.lr,weight_decay=args.wd)
    # schedulers = []
    # steps = [10*i for i in range(1, 100)]
    # for op in [optimQ, *optimW]:
    #     schedulers.append(utils.CyclicCosAnnealingLR(op, steps, eta_min=1e-8))

    best_test_acc, best_test_loss, = 0., np.inf
    # best_test_adv_acc, best_test_adv_loss, = 0., np.inf
    

    '''prepare data'''

    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Adjust color
        transforms.RandomRotation(degrees=10),  # Randomly rotate the image up to 10 degrees
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])

    # Compose the transformations for both resizing and augmentation
    transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        augmentation_transform,
    ])


    # Specify dataset path
    dataset_path = 'feature_extractor_dataset'

    # Create a dataset using ImageFolder
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)

    # Define the proportions for training and test sets
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Access the classes attribute directly from the full_dataset
    num_classes = len(full_dataset.classes)
    print(num_classes)

    # Create DataLoader for training and test sets
    trainset = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4)
    testset = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4)
      
    print ('==> Begin Training')
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.epochs):
            acc = 0.
            for batch_idx, (data, target) in enumerate(trainset):

                data = data.to(args.device)
                target = target.to(args.device)

                # s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.batch_size, args.s)).to(args.device)
                s = torch.empty(args.batch_size,args.s).normal_(mean=args.s_mean,std=args.s_std).to(args.device)
                codes = mixer(s)
                codes = codes.view(args.batch_size, args.ngen, args.z)
                codes = torch.stack([codes[:, i] for i in range(args.ngen)])
                params = generator(codes)

                '''diversity loss '''
                var = torch.var(torch.flatten(params[0],start_dim=1,end_dim=-1),dim=0).mean()+torch.var(torch.flatten(params[2],start_dim=1,end_dim=-1),dim=0).mean()+torch.var(torch.flatten(params[4],start_dim=1,end_dim=-1),dim=0).mean()
                varloss = args.diversity_lambda*torch.exp(-var)
                '''accuracy loss'''
                clf_loss = 0.
                i = 0
                train_acc = [0]* args.batch_size
                for (layers) in zip(*params):
                    out = hypergan.eval_f(args, layers, data)
                    loss = F.cross_entropy(out, target)
                    pred = out.data.max(1, keepdim=True)[1]
                    acc += pred.eq(target.data.view_as(pred)).long().cpu().sum()
                    train_acc[i] += pred.eq(target.data.view_as(pred)).long().cpu().sum()
                    clf_loss += loss
                    i+=1

                """ calculate total loss on Q and G """
                Q_loss = max(varloss - args.threshold,0)
                G_loss = clf_loss / args.batch_size
                QGA_loss = Q_loss  + G_loss
                # QGA_loss = G_loss
                QGA_loss.backward(retain_graph=True)
                
                optimQ.step()
                optimF.step()
                for optim in optimW:
                    optim.step()

                optimQ.zero_grad()
                optimF.zero_grad()
                for optim in optimW:
                    optim.zero_grad()
            
                '''update attack model to min adv loss'''
                params = [p.detach() for p in params] 
                # print(train_acc)

            acc /= len(trainset.dataset)*args.batch_size
        
            """ print training accuracy """
            print ('**************************************')
            print ('Epoch: {}'.format(epoch))
            print('Train ACC: {}, G Loss: {}, Q Loss: {}'.format(acc,G_loss, Q_loss))
            print ('best test loss: {}'.format(best_test_loss))
            print ('best test acc: {}'.format(best_test_acc))
            print ('**************************************')

            """ test random draw on testing set """
            test_acc = 0.
            test_acc_list = [0]* args.batch_size
            test_loss = 0.

            with torch.no_grad():
                for i, (data, target) in enumerate(testset):

                    data = data.to(args.device)
                    target = target.to(args.device)

                    s = torch.normal(mean=args.s_mean, std=args.s_std, size=(args.batch_size, args.s)).to(args.device)
                    codes = mixer(s)
                    codes = codes.view(args.batch_size, args.ngen, args.z)
                    codes = torch.stack([codes[:, i] for i in range(args.ngen)])
                    params = generator(codes)
                    
                    idx = 0
                    for (layers) in zip(*params):
                        out = hypergan.eval_f(args, layers, data)

                        loss = F.cross_entropy(out, target)

                        test_loss += loss.item()

                        pred = out.data.max(1, keepdim=True)[1]

                        test_acc += pred.eq(target.data.view_as(pred)).long().cpu().sum()
                        test_acc_list[idx] +=pred.eq(target.data.view_as(pred)).long().cpu().sum()
                        idx+=1

                test_loss /= len(testset.dataset) * args.batch_size
                test_acc /= len(testset.dataset) * args.batch_size

                print ('Test Accuracy: {}, Test Loss: {}'.format(test_acc, test_loss))
                # print('Test List Accuracy: {}'.format(test_acc_list))
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                if epoch >= 500 and epoch % 100 == 0:
                    hypergan.save_models(args, test_acc, epoch)
                    best_test_acc = test_acc


if __name__ == '__main__':
    args = load_args()
    train(args)
