## python adversarial_training.py --batch_size 4
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import compute_batch
import numpy as np
import os
import torch.nn.functional as F
from torch.utils.data import random_split
import inversion_net
from PIL import Image
import argparse

def main(args=None):
    IMAGE_SIZE = (256, 256)
    print("ok")

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Specify dataset path
    dataset_path = 'training_dataset'

    # Create a dataset using ImageFolder
    # In (C,H,W) format
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)
    

    # Access the classes attribute directly from the full_dataset
    num_classes = len(full_dataset.classes)

    # Create DataLoader for training and test sets
    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
    num_epochs = 50

    for idx, item in enumerate(os.listdir(args.model_path)):
        model_path = os.path.join(args.model_path,item)
        extractor = compute_batch.load_extractor(device,model_path,args.model_type)
        model = inversion_net.ImageRestorationCNN().to(device)
        # Define the MSE loss function
        criterion = nn.MSELoss()
        # Define the optimizer (e.g., Adam)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print("load")
        for epoch in range(num_epochs):
            for batch_idx, (image, labels) in enumerate(train_loader):
                try:
                    original_images = torch.tensor(image).float().to(device)
                    original_images = original_images * 255.0
                    protected_images = compute_batch.compute(original_images, device, extractor)
                    final_images_np = torch.tensor(protected_images).float().to(device)
        
                    # Perform inference using your model
                    inputs = final_images_np / 255.0
                    outputs = model(inputs) * 255.0
                    outputs = nn.functional.interpolate(outputs, size=(final_images_np.size(2), final_images_np.size(3)), mode='nearest')
                    outputs += final_images_np
                    mse_loss = criterion(original_images, outputs)
                    mse_loss_f = criterion(original_images, final_images_np)
                    
                    bottleneck_o = extractor(original_images)
                    bottleneck_i = extractor(outputs)
                    bottleneck_m = extractor(final_images_np)
                    
                    # Calculate mean squared error loss using PyTorch
                    feature_space_loss = F.mse_loss(bottleneck_o, bottleneck_i)
                    feature_space_loss_m = F.mse_loss(bottleneck_o, bottleneck_m)
                    
                    t_loss = mse_loss
                    
                    print("Epoch:", epoch + 1, "Batch:", batch_idx + 1)
                    print(f'MSE Loss Final: {mse_loss_f.cpu().detach().numpy()}, Feature Space Loss: {feature_space_loss_m.cpu().detach().numpy()}')
                    print(f'MSE Loss: {mse_loss.detach().cpu().detach().numpy()}, Feature Space Loss: {feature_space_loss.cpu().detach().numpy()}')
        
                    optimizer.zero_grad()
                    t_loss.backward()
                    optimizer.step()  
                    
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            
                        
            save_path = args.save_path + '/' + 'invesion' + str(idx) + '.pth' 
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train inversion model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--model', type=int, default=0, help='use model to train')
    parser.add_argument('--model_type', type=str, default='feature_model_med_1', help='model type')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('--save_path', type=str, default=None, help='save model path')

    args = parser.parse_args()
 
    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")


