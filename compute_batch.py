import numpy as np
import os
import sys
import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision import transforms
from torchvision.models import resnet18
import feature_model_med_1

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

IMAGE_SHAPE = (3,256,256)
LEARNING_RATE = 0.01
INITIAL_CONST = 1e2
# MAX_ITERATIONS = 40
MAX_ITERATIONS = 40
TANH_CONSTANT = 2 - 1e-6
INTENSITY_RANGE = 'raw'
L_THRESHOLD = 0.1

class Extractor(nn.Module):
    def __init__(self, model):
        super(Extractor, self).__init__()
        self.model = model

    def forward(self, x):
        x = x / 255.0
        embeds = l2_norm(self.model(x))
        return embeds
    
def l2_norm(x, axis=1):
    """l2 norm"""
    norm = torch.norm(x, dim=axis, keepdim=True)
    output = x / norm
    return output


def load_extractor(device,model_path,model_type):

    module_name = model_type
    feature_model = importlib.import_module(model_type)

    model = feature_model.FaceRecognitionModel(num_classes=20)
    # Load the state dict from the specified path
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Initialize the model with the state dict
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")

    except Exception as e:
        print("Error loading model:", e)
        
    feature_extractor = feature_model.FaceRecognitionModelFeatureExtractor(num_classes=20)  # Define a new model with only the feature extractor
    feature_extractor.load_state_dict(model.state_dict())
    # Copy the weights from the loaded model to the new model's feature extractor
    # new_model.features = model.features
    feature_extractor = Extractor(feature_extractor).to(device)
    return feature_extractor


def resize_tensor(input_tensor, model_input_shape):
    # Check if the input tensor has the same shape as the model input shape
    if input_tensor.shape[1:] == model_input_shape or model_input_shape[1] is None:
        return input_tensor

    # Resize the tensor using PyTorch's interpolate function
    resized_tensor = F.interpolate(input_tensor, size=model_input_shape[:2], mode='bilinear', align_corners=False)
    
    return resized_tensor

def preprocess_arctanh(imgs):
    """ Do tan preprocess """
    imgs = imgs / 255.0
    imgs = imgs - 0.5
    imgs = imgs * TANH_CONSTANT
    tanh_imgs = np.arctanh(imgs)
    return tanh_imgs

def reverse_arctanh(imgs):
    raw_img = (torch.tanh(imgs) / TANH_CONSTANT + 0.5) * 255
    return raw_img

def clipping(imgs):
    imgs = imgs.detach().cpu().numpy()
    imgs = np.clip(imgs, 0, 255.0)
    return imgs

def calc_dissim(source_raw, source_mod_raw, l_threshold):
    batch_size = source_raw.size(0)

    ssim_values = torch.zeros(batch_size)
    msssim_split = 1 - ssim(source_mod_raw, source_raw,size_average = False)
    
    dist_raw = msssim_split / 2.0
    dist = torch.maximum(dist_raw - l_threshold, torch.tensor(0.0))

    dist_raw_avg = torch.mean(dist_raw)
    dist_sum = torch.sum(dist)
    return dist, dist_raw, dist_sum, dist_raw_avg

def calc_bottlesim(source_input, original_input, extractor):
    bottleneck_a = extractor(source_input)
    bottleneck_s = extractor(original_input)
    bottleneck_diff = bottleneck_a - bottleneck_s
    cur_bottlesim = torch.sum(torch.abs(bottleneck_diff), dim=1)
    # cur_bottlesim = cur_bottlesim / scale_factor
    # print("cur_bottlesim",cur_bottlesim)
    return cur_bottlesim, torch.sum(cur_bottlesim)

def compute_feature_loss(aimg_raw, simg_raw, aimg_input, simg_input, const, const_diff, extractor):
    input_space_loss, dist_raw, input_space_loss_sum, input_space_loss_raw_avg = calc_dissim(aimg_raw, simg_raw, L_THRESHOLD)
    
    feature_space_loss, feature_space_loss_sum = calc_bottlesim(aimg_input, simg_input, extractor)
    loss = const * torch.square(input_space_loss) - feature_space_loss * const_diff
    # print("input space loss :",const * torch.square(input_space_loss))
    # print("feature space loss :",feature_space_loss * const_diff)
    loss_sum = torch.sum(loss)
    # print("loss :",loss_sum)
    return loss_sum, feature_space_loss, input_space_loss_raw_avg, dist_raw

def compute(source_imgs, device,extractor):
    nb_imgs = source_imgs.shape[0]
    source_imgs = np.array(source_imgs.cpu(), dtype=np.float32)
    best_bottlesim = [0] * nb_imgs
    best_adv = np.zeros(source_imgs.shape)
    # modifier = torch.nn.Parameter((2 * torch.rand(tuple([len(source_imgs)] + list(IMAGE_SHAPE))) - 1) * 1e-4, requires_grad=True,device=device)
    modifier = (2 * torch.rand(tuple([len(source_imgs)] + list(IMAGE_SHAPE))) - 1) * 1e-4
    modifier = modifier.to(device)
    modifier.requires_grad = True  # If you want to compute gradients for this tensor
    optimizer = torch.optim.Adam([modifier], lr=LEARNING_RATE)

    simg_tanh = preprocess_arctanh(source_imgs)
    # Initialize the modifier parameter with the same size
    
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    const = torch.tensor(np.ones(len(source_imgs)) * INITIAL_CONST, dtype=torch.float32).to(device)
    const_diff = torch.tensor(np.ones(len(source_imgs)) * 1.0, dtype=torch.float32).to(device)
    simg_raw = torch.tensor(source_imgs, dtype=torch.float32, requires_grad=True).to(device)
    simg_tanh = torch.tensor(simg_tanh, dtype=torch.float32, requires_grad=True).to(device)
    
    outside_list = np.ones(len(source_imgs))
    it = 0

    while it < MAX_ITERATIONS:
        it += 1
        optimizer.zero_grad()
        aimg_raw = reverse_arctanh(simg_tanh + modifier)
        actual_modifier = aimg_raw - simg_raw
        actual_modifier = torch.clamp(actual_modifier, -15.0, 15.0)
        aimg_raw = simg_raw + actual_modifier
        simg_raw = reverse_arctanh(simg_tanh)
        aimg_input = aimg_raw
        simg_input = simg_raw
        
        loss, internal_dist, input_dist_avg, dist_raw = compute_feature_loss(
            aimg_raw, simg_raw, aimg_input, simg_input, const, const_diff,extractor)
        if it == 1:
            loss.backward(retain_graph=True)
            optimizer.step()
            grad = torch.autograd.grad(loss, modifier, create_graph=True)[0]
            # Update modifier manually
            modifier.data = modifier - torch.sign(grad) * 0.01
            
        else:
            loss.backward(retain_graph=True)
            optimizer.step()
            
        # print("Actual modifier", actual_modifier)
        # for e, (input_dist, feature_d, mod_img) in enumerate(zip(dist_raw, internal_dist, aimg_input)):
        #     if e >= nb_imgs:
        #         break     
        #     # Convert to NumPy for condition checking
        #     input_dist_np = input_dist.cpu().detach().numpy()
        #     feature_d_np = feature_d.cpu().detach().numpy()
        #     const_diff_np = const_diff[e].cpu().detach().item()  # Convert to scalar value
        
        #     if input_dist_np <= L_THRESHOLD * 0.9 and const_diff_np <= 129:
        #         const_diff[e] *= 2
        #         if outside_list[e] == -1:
        #             const_diff[e] = 1
        #         outside_list[e] = 1
        #     elif input_dist_np >= L_THRESHOLD * 1.1 and const_diff_np >= 1 / 129:
        #         const_diff[e] /= 2
        #         if outside_list[e] == 1:
        #             const_diff[e] = 1
        #         outside_list[e] = -1
        #     else:
        #         const_diff[e] = 1.0
        #         outside_list[e] = 0
        
        #     if input_dist_np <= L_THRESHOLD * 1.1 and feature_d_np > best_bottlesim[e]:
        #         best_bottlesim[e] = feature_d_np
        #         best_adv[e] = mod_img.cpu().detach().numpy()
        
    
    # for e, diff in enumerate(best_bottlesim):
    #     if diff.item() < 0.3 and dist_raw[e].item() < 0.015 and internal_dist[e].item() > diff.item():
    #         best_adv[e] = aimg_input[e].cpu().detach().numpy()

    # for e, diff in enumerate(best_bottlesim):
    #     if internal_dist[e].item() > diff.item():
    #         best_adv[e] = aimg_input[e].cpu().detach().numpy()

    best_adv = aimg_raw
    best_adv = clipping(best_adv[:nb_imgs])
    return best_adv
