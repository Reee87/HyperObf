import inversion_net
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import compute_batch
import os
import torch.nn as nn
from PIL import Image
import argparse


def main(args=None):
    dataset_path_root = 'users/train'
    
    BATCH = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inv_model_path = args.inversion_model_path
    restore_model = inversion_net.ImageRestorationCNN()

    # Load the state dict from the specified path
    state_dict = torch.load(inv_model_path, map_location=device)
    # Initialize the model with the state dict
    restore_model.load_state_dict(state_dict)
    restore_model = restore_model.to(device)

    save_root_cloaked = os.path.join(args.save_file_root,"cloaked/")
    save_root_restored = os.path.join(args.save_file_root,"restored/")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    for item in os.listdir(dataset_path_root):
        item_path = os.path.join(dataset_path_root, item)
        extractor = compute_batch.load_extractor(device,args.model_path,args.model_type)
        full_dataset = datasets.ImageFolder(item_path, transform=transform)
        train_loader = DataLoader(full_dataset, batch_size=BATCH, shuffle=False)

        for batch_idx, (image, labels) in enumerate(train_loader):
            original_images = torch.tensor(image).float().to(device)
            original_images = original_images * 255.0
            protected_images = compute_batch.compute(original_images,device,extractor)
            final_images_np = torch.tensor(protected_images).float().to(device)
            # original_images = image.clone().detach().requires_grad_(True).to(device)
            outputs = restore_model(final_images_np / 255.0) * 255.0
            raw_images = final_images_np
            outputs = nn.functional.interpolate(outputs, size=(raw_images.size(2), raw_images.size(3)), mode='nearest')
            outputs += raw_images

            for i in range(len(original_images)):
                # Convert PyTorch tensor to NumPy array
                p_img_np = torch.clamp(final_images_np[i], 0, 255).detach().cpu().numpy().astype(np.uint8)
                r_img_np = torch.clamp(outputs[i], 0, 255).detach().cpu().numpy().astype(np.uint8)

                # Create a PIL Image from the NumPy array
                image_pil = Image.fromarray(np.transpose(p_img_np, (1, 2, 0)))  # Assuming outputs are in (C, H, W) format
                image_ril = Image.fromarray(np.transpose(r_img_np, (1, 2, 0)))

                # Save the image using PIL
                # path = full_dataset.classes[labels[i]]  # Assuming labels are folder names
                
                file_name = save_root_cloaked + f"{item}/1/{i + batch_idx * BATCH}.png"
                directory = os.path.dirname(file_name)
                # Check if the directory exists, and create it if not
                if not os.path.exists(directory):
                    os.makedirs(directory)
                image_pil.save(file_name)

                file_name = save_root_restored + f"{item}/1/{i + batch_idx * BATCH}.png"
                directory = os.path.dirname(file_name)
                # Check if the directory exists, and create it if not
                if not os.path.exists(directory):
                    os.makedirs(directory)
                image_ril.save(file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train inversion model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--model', type=int, default=0, help='use model to train')
    parser.add_argument('--model_type', type=str, default='feature_model_med_1', help='model type')
    parser.add_argument('--model_path', type=str, default='MaskNet/scratch/model.pth', help='model path')  
    parser.add_argument('--inversion_model_path', type=str, default='InvNet/scratch/inversion_model.pth', help='inversion model path')
    parser.add_argument('--save_file_root', type=str, default='Images/scratch', help='save path')


    args = parser.parse_args()
 
    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")