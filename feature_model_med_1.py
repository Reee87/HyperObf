import torch
import torch.nn as nn
import torch.nn.functional as F
from hypergan_base import HyperGAN_Base
import os
    

class FaceRecognitionModel(nn.Module):
    def __init__(self,args= None,num_classes=20):
        super(FaceRecognitionModel, self).__init__()

        self.num_classes = int(num_classes)
        self.after_conv_size = int((256/16)**2*32)
        self.in_channels = 3
        
        self.conv1 = nn.Conv2d(self.in_channels, 16, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(16, 64, 3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(64, 16, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.linear1 = nn.Linear(self.after_conv_size, 256)
        # self.linear2 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 4)
        # x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        # x = x.view(-1,self.after_conv_size)
        x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        x = self.fc2(x)
        return x

class FaceRecognitionModelFeatureExtractor(FaceRecognitionModel):
    def __init__(self, num_classes=20):
        super(FaceRecognitionModelFeatureExtractor, self).__init__(num_classes)

    def forward(self, x):
        x = super(FaceRecognitionModelFeatureExtractor, self).forward(x)
        # Remove the output of the last fully connected layer
        return x[:, :-1]

class Mixer(nn.Module):
    def __init__(self, args):
        super(Mixer, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.s, 256, bias=self.bias)
        self.linear2 = nn.Linear(256, 256, bias=self.bias)
        self.linear3 = nn.Linear(256, self.z*self.ngen, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.view(-1, self.s) #flatten filter size
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        return x
        # x = x.view(-1, self.ngen, self.z)
        # w = torch.stack([x[:, i] for i in range(self.ngen)])
        # return w


class GeneratorW1(nn.Module):
    def __init__(self, args):
        super(GeneratorW1, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 256, bias=self.bias)
        self.linear2 = nn.Linear(256, 256, bias=self.bias)
        self.linear3 = nn.Linear(256, 16*3*3*3 + 16, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :16*3*3*3], x[:, -16:]
        w = w.view(-1, 16, 3, 3, 3)
        b = b.view(-1, 16)
        return (w, b)
    

class GeneratorW4(nn.Module):
    def __init__(self, args):
        super(GeneratorW4, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 256, bias=self.bias)
        self.linear2 = nn.Linear(256, 256, bias=self.bias)
        self.linear3 = nn.Linear(256, 16*32*3*3+32, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :16*32*3*3], x[:, -32:]
        w = w.view(-1, 32, 16, 3, 3)
        b = b.view(-1, 32)
        return (w, b)


class GeneratorW5(nn.Module):
    def __init__(self, args):
        super(GeneratorW5, self).__init__()
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.linear1 = nn.Linear(self.z, 256, bias=self.bias)
        self.linear2 = nn.Linear(256, 256, bias=self.bias)
        self.linear3 = nn.Linear(256, 16*16*32*256+256, bias=self.bias)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        if not self.bias:
            self.bn1.bias.data.zero_()
            self.bn2.bias.data.zero_()
        x = torch.zeros_like(x).normal_(0, 0.01) + x
        x = F.relu(self.bn1(self.linear1(x)))
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.linear3(x)
        w, b = x[:, :256 * 16 * 16 *32], x[:, -256:]
        w = w.view(-1, 256, 16 * 16 *32)
        b = b.view(-1, 256)
        return (w, b)


# class GeneratorW6(nn.Module):
#     def __init__(self, args):
#         super(GeneratorW6, self).__init__()
#         for k, v in vars(args).items():
#             setattr(self, k, v)
#         self.num_classes = int(args.num_classes)
#         self.linear1 = nn.Linear(self.z, 256, bias=self.bias)
#         self.linear2 = nn.Linear(256, 256, bias=self.bias)
#         self.linear3 = nn.Linear(256, 256*256+256, bias=self.bias)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.bn2 = nn.BatchNorm1d(256)

#     def forward(self, x):
#         if not self.bias:
#             self.bn1.bias.data.zero_()
#             self.bn2.bias.data.zero_()
#         x = torch.zeros_like(x).normal_(0, 0.01) + x
#         x = F.relu(self.bn1(self.linear1(x)))
#         x = F.relu(self.bn2(self.linear2(x)))
#         x = self.linear3(x)
#         w, b = x[:, :256*256], x[:, -256:]
#         w = w.view(-1, 256, 256)
#         b = b.view(-1, 256)
#         return (w, b)



class HyperGAN(HyperGAN_Base):
    
    def __init__(self, args):
        super(HyperGAN, self).__init__(args)
        self.mixer = Mixer(args).to(args.device)
        self.generator = self.Generator(args)
        self.FC = nn.Linear(256, args.num_classes).to(args.device)
        self.model = FaceRecognitionModel(args).to(args.device)
        self.feature = FaceRecognitionModelFeatureExtractor(args).to(args.device)

    class Generator(object):
        def __init__(self, args):
            self.W1 = GeneratorW1(args).to(args.device)
            # self.W2 = GeneratorW2(args).to(args.device)
            # self.W3 = GeneratorW3(args).to(args.device)
            self.W4 = GeneratorW4(args).to(args.device)
            self.W5 = GeneratorW5(args).to(args.device)
            # self.W6 = GeneratorW6(args).to(args.device)

        def __call__(self, x):
            w1, b1 = self.W1(x[0])
            # w2, b2 = self.W2(x[1])
            # w3, b3 = self.W3(x[2])
            w4, b4 = self.W4(x[1])
            w5, b5 = self.W5(x[2])
            # w6, b6 = self.W6(x[3])
            # layers = [w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, w6, b6]
            layers = [w1, b1, w4, b4, w5, b5]
            return layers
        
        def as_list(self):
            return [self.W1,self.W4, self.W5]

    """ functional model for training """
    def eval_f(self, args, Z, data):
        w1, b1, w4, b4, w5, b5 = Z
        # w1, b1, w4, b4, w5, b5 = Z
        x = F.leaky_relu(F.conv2d(data, w1,stride=1, padding=1, bias=b1))
        # x = F.relu(F.conv2d(x, w2, stride=1, padding=1, bias=b2))
        x = F.max_pool2d(x,4)
        # x = F.relu(F.conv2d(x, w3, stride=1, padding=1, bias=b3))
        x = F.leaky_relu(F.conv2d(x, w4, stride=1, padding=1, bias=b4))
        x = F.max_pool2d(x, 4)    
        x = x.view(-1,32*16*16)
        x = F.linear(x, w5, bias=b5)
        x = F.leaky_relu(x)
        # x = F.linear(x, w6, bias=b6)
        # x = F.leaky_relu(x)
        x = self.FC(x)
        return x
    
    def extract(self,Z,model=None):
        w1, b1, w4, b4, w5, b5 = Z



    def restore_models(self, args):
        save_dir = f"saved_models/hypernet/model_1/"
        load_path = os.path.join(save_dir, f"epoch_{args.epochs}_div_{args.diversity_lambda}_thre_{args.threshold}.pth")
        d = torch.load(load_path, map_location=args.device)
        self.mixer.load_state_dict(d['mixer']['state_dict'])
        self.mixer.to(args.device)
        generators = self.generator.as_list()

        generators[0].load_state_dict(d['W1']['state_dict'])
        generators[1].load_state_dict(d['W4']['state_dict'])
        generators[2].load_state_dict(d['W5']['state_dict'])
        # generators[3].load_state_dict(d['W6']['state_dict'])
        self.FC.load_state_dict(d['FC']['state_dict'])
        self.FC.to(args.device)


    def save_models(self, args, metrics=None,epoch=None):
        save_dict = {
                'mixer': {'state_dict': self.mixer.state_dict()},
                'W1': {'state_dict': self.generator.W1.state_dict()},
                # 'W2': {'state_dict': self.generator.W2.state_dict()},
                # 'W3': {'state_dict': self.generator.W3.state_dict()},
                'W4': {'state_dict': self.generator.W4.state_dict()},
                'W5': {'state_dict': self.generator.W5.state_dict()},
                # 'W6': {'state_dict': self.generator.W6.state_dict()},
                'FC': {'state_dict': self.FC.state_dict()}
                }
        save_dir = f"saved_models/hypernet/model_1/"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(save_dict, os.path.join(save_dir, f"epoch_{epoch}_div_{args.diversity_lambda}_thre_{args.threshold}.pth"))
