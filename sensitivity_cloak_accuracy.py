import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import ConcatDataset,Dataset
from torch.utils.data import random_split
from sklearn.metrics import confusion_matrix, classification_report
import compute_batch
from PIL import Image
import numpy as np
import re
import os
import argparse


class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def main(args=None):
    # Set device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Specify dataset path
    train_dataset_path = 'training_dataset'
    test_dataset_path = 'test_dataset'

    dataset_path_root = 'users/train'

    # Create a dataset using ImageFolder
    train_dataset = datasets.ImageFolder(train_dataset_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_dataset_path, transform=transform)

    fixed_label = 50
    matrix = [[[0 for _ in range(10)] for _ in range(10)] for _ in range(5)]

    # parameters_set = set()

    # with open("output/sensitivity_variance.txt", "r") as f:
    #     for line in f:
    #         parameter_value = line.split()[6]
    #         parameters_set.add(parameter_value)

    with open('output/sensitivity_variance.txt', 'r') as file:
        data = file.readlines()

    parameters_set = set()

    pattern = r'parameters (\w+\.\w+)'

    for line in data:
        match = re.search(pattern, line)
        if match:
            parameters_set.add(match.group(1))

    print(parameters_set)

    try:
        for para in os.listdir(args.root_path):
            print("para: " + para)
            if para in parameters_set:
                print("continue")
                continue
            para_path = os.path.join(args.root_path,para)
            
            mean = [0.0] * 5
            variance = [0.0] * 5
            para_accuracy = [[0.0 for _ in range(5)] for _ in range(5)]

            for idx, m_idx in enumerate(os.listdir(para_path)):
                if idx >= 5:
                    break

                model_path = os.path.join(para_path,m_idx)
                extractor = compute_batch.load_extractor(device,model_path,'feature_model_med_1')

                for item in os.listdir(dataset_path_root):
                    target_test = "users/test"
                    target_test = os.path.join(target_test, item)
                    item_path = os.path.join(dataset_path_root, item)
                    full_dataset = datasets.ImageFolder(item_path, transform=transform)
                    train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)

                    for batch_idx, (image, labels) in enumerate(train_loader):
                        original_images = torch.tensor(image).float().to(device)
                        original_images = original_images * 255.0
                        protected_images = compute_batch.compute(original_images,device,extractor)
                        final_images_np = torch.tensor(protected_images).float().to(device)
                        # original_images = image.clone().detach().requires_grad_(True).to(device)

                        for i in range(len(original_images)):
                            # Convert PyTorch tensor to NumPy array
                            p_img_np = torch.clamp(final_images_np[i], 0, 255).detach().cpu().numpy().astype(np.uint8)

                            # Create a PIL Image from the NumPy array
                            image_pil = Image.fromarray(np.transpose(p_img_np, (1, 2, 0)))  # Assuming outputs are in (C, H, W) format
                            # Save the image using PIL
                            # path = full_dataset.classes[labels[i]]  # Assuming labels are folder names
                            
                            file_name = 'Images/sensitivities' + f"/{para}/{m_idx}/{item}/1/{i + batch_idx * args.batch_size}.png"
                            directory = os.path.dirname(file_name)
                            # Check if the directory exists, and create it if not
                            if not os.path.exists(directory):
                                os.makedirs(directory)
                            image_pil.save(file_name)

                    target_train = 'Images/sensitivities' + f"/{para}/{m_idx}/{item}"

                    target_test_dataset = datasets.ImageFolder(target_test, transform=transform,target_transform= lambda x: fixed_label)
                    target_train_dataset = datasets.ImageFolder(target_train, transform=transform,target_transform= lambda x: fixed_label)
                    
                    combined_train_dataset = ConcatDataset([train_dataset,target_train_dataset])
                    combined_test_dataset = ConcatDataset([test_dataset,target_test_dataset])

                    new_train_classes = list(set(train_dataset.classes) |set(target_train_dataset.classes)) 
                    new_test_classes = list(set(test_dataset.classes) | set(target_test_dataset.classes))

                    # combined_train_dataset.classes = new_train_classes
                    # combined_test_dataset.classes = new_test_classes
                    num_classes = len(new_test_classes)

                    full_train_loader = DataLoader(combined_train_dataset, batch_size=16, shuffle=True, num_workers=4)
                    full_test_loader = DataLoader(combined_test_dataset, batch_size=16, shuffle=True, num_workers=4)

                    # Define a simple face recognition model (you can customize the architecture)

                    # Instantiate the model and move it to the device
                    model = FaceRecognitionModel(num_classes=num_classes).to(device)

                    # Define loss function and optimizer
                    criterion = nn.CrossEntropyLoss()

                    optimizer = optim.Adam(model.parameters(), lr=0.0001) 

                    # Training loop
                    num_epochs = 10
                    for epoch in range(num_epochs):
                        model.train()
                        running_loss = 0.0

                        for inputs, labels in full_train_loader:
                            inputs, labels = inputs.to(device), labels.to(device)

                            optimizer.zero_grad()
                            outputs = model(inputs)
                            loss = criterion(outputs, labels) 
                            loss.backward()
                            optimizer.step()

                            running_loss += loss.item()

                        # Calculate and print training accuracy
                        model.eval()
                        correct_train = 0
                        total_train = 0

                        with torch.no_grad():
                            for inputs, labels in full_train_loader:
                                inputs, labels = inputs.to(device), labels.to(device)

                                outputs = model(inputs)
                                _, predicted = torch.max(outputs.data, 1)
                                total_train += labels.size(0)
                                correct_train += (predicted == labels).sum().item()

                        accuracy_train = correct_train / total_train

                        # Test the model
                        model.eval()
                        correct_test = 0
                        total_test = 0
                        
                        all_labels = []
                        all_predicted = []

                        class_of_interest = 50
                        correct_class = 0
                        total_class = 0

                        with torch.no_grad():
                            for inputs, labels in full_test_loader:
                                inputs, labels = inputs.to(device), labels.to(device)

                                outputs = model(inputs)
                                # print("test output",outputs)
                                _, predicted = torch.max(outputs.data, 1)
                                total_test += labels.size(0)
                                correct_test += (predicted == labels).sum().item()

                                # Calculate accuracy for the specific class
                                mask = labels == class_of_interest
                                total_class += mask.sum().item()
                                correct_class += ((predicted == labels) & mask).sum().item()
                                
                                all_labels.extend(labels.cpu().numpy())
                                all_predicted.extend(predicted.cpu().numpy())

                        accuracy_test = correct_test / total_test
                        accuracy_class = correct_class / total_class

                        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}")
                        print(f"Accuracy for Class {class_of_interest}: {accuracy_class}")

                        if epoch == 9:
                            para_accuracy[int(item)][idx]=accuracy_class
                            print(f"Accuracy for para {para} model {idx} item {item}: {accuracy_class}")
            
            for i in range(len(para_accuracy)):
                row_data = para_accuracy[i]
                mean[i] = np.mean(row_data)
                variance[i] = np.var(row_data)

            print("mean value:", mean)
            print("variance value:", variance)

            with open(f"output/sensitivity_mean.txt", 'a') as f:
                f.write(f"\nThe mean value of the parameters {para} is: ")
                f.write(", ".join(map(str, mean)))

            with open(f"output/sensitivity_variance.txt", 'a') as f:
                f.write(f"\nThe variance value of the parameters {para} is: ")
                f.write(", ".join(map(str, variance)))
             
    except Exception as e:
        print("error", e)
                

        # for item in os.listdir(test_path):
        #     target_test = "users/test"
        #     target_train = os.path.join(test_path, item)
        #     print(target_train)
        #     target_test = os.path.join(target_test, item)
        #     print(target_test)
        #     target_test_dataset = datasets.ImageFolder(target_test, transform=transform,target_transform= lambda x: fixed_label)
        #     target_train_dataset = datasets.ImageFolder(target_train, transform=transform,target_transform= lambda x: fixed_label)

        #     combined_train_dataset = ConcatDataset([train_dataset,target_train_dataset])
        #     combined_test_dataset = ConcatDataset([test_dataset,target_test_dataset])

        #     new_train_classes = list(set(train_dataset.classes) |set(target_train_dataset.classes)) 
        #     new_test_classes = list(set(test_dataset.classes) | set(target_test_dataset.classes))

        #     # combined_train_dataset.classes = new_train_classes
        #     # combined_test_dataset.classes = new_test_classes
        #     num_classes = len(new_test_classes)

        #     train_loader = DataLoader(combined_train_dataset, batch_size=16, shuffle=True, num_workers=4)
        #     test_loader = DataLoader(combined_test_dataset, batch_size=16, shuffle=True, num_workers=4)

        #     # Define a simple face recognition model (you can customize the architecture)

        #     # Instantiate the model and move it to the device
        #     model = FaceRecognitionModel(num_classes=num_classes).to(device)

        #     # Define loss function and optimizer
        #     criterion = nn.CrossEntropyLoss()

        #     optimizer = optim.Adam(model.parameters(), lr=0.0001) 

        #     # Training loop
        #     num_epochs = 30
        #     for epoch in range(num_epochs):
        #         model.train()
        #         running_loss = 0.0

        #         for inputs, labels in train_loader:
        #             inputs, labels = inputs.to(device), labels.to(device)

        #             optimizer.zero_grad()
        #             outputs = model(inputs)
        #             loss = criterion(outputs, labels) 
        #             loss.backward()
        #             optimizer.step()

        #             running_loss += loss.item()

        #         # Calculate and print training accuracy
        #         model.eval()
        #         correct_train = 0
        #         total_train = 0

        #         with torch.no_grad():
        #             for inputs, labels in train_loader:
        #                 inputs, labels = inputs.to(device), labels.to(device)

        #                 outputs = model(inputs)
        #                 _, predicted = torch.max(outputs.data, 1)
        #                 total_train += labels.size(0)
        #                 correct_train += (predicted == labels).sum().item()

        #         accuracy_train = correct_train / total_train

        #         # Test the model
        #         model.eval()
        #         correct_test = 0
        #         total_test = 0
                
        #         all_labels = []
        #         all_predicted = []

        #         class_of_interest = 49
        #         correct_class = 0
        #         total_class = 0

        #         with torch.no_grad():
        #             for inputs, labels in test_loader:
        #                 inputs, labels = inputs.to(device), labels.to(device)

        #                 outputs = model(inputs)
        #                 # print("test output",outputs)
        #                 _, predicted = torch.max(outputs.data, 1)
        #                 total_test += labels.size(0)
        #                 correct_test += (predicted == labels).sum().item()

        #                 # Calculate accuracy for the specific class
        #                 mask = labels == class_of_interest
        #                 total_class += mask.sum().item()
        #                 correct_class += ((predicted == labels) & mask).sum().item()
                        
        #                 all_labels.extend(labels.cpu().numpy())
        #                 all_predicted.extend(predicted.cpu().numpy())

        #         accuracy_test = correct_test / total_test
        #         accuracy_class = correct_class / total_class

        #         # Print training and test accuracy
        #         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}")
        #         print(f"Accuracy for Class {class_of_interest}: {accuracy_class}")

        #         if epoch == 29:
        #             matrix[item][masknet][invnet]=accuracy_class
        #     # Instantiate the model
            
        #     # conf_matrix = confusion_matrix(all_labels, all_predicted)
        #     # class_report = classification_report(all_labels, all_predicted, target_names=test_dataset.classes)

        #     # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}")
        #     # print("Confusion Matrix:")
        #     # print(conf_matrix)
        #     # conf_matrix_str = str(confusion_matrix.numpy())

        #     # Write the string to a file
        #     # with open('matrix.txt', 'w') as f:
        #     #     f.write(conf_matrix_str)
        #     # print("Classification Report:")
        #     # print(class_report)
            
            
        #     # model_path = 'test_model_resnet18.pth'
        #     # torch.save(model.state_dict(), model_path)

        #     model.train()  # Set the model back to training mode for the next epoch
        
        #     output_dir = f"output/{args.root_path}"
        #     os.makedirs(output_dir, exist_ok=True)
        #     with open(f"output/{args.root_path}/output.txt", "w") as file:
        #         for i in range(5):
        #             for row in matrix[i]:                 
        #                 line = ' '.join(map(str, row))                  
        #                 file.write(line + '\n')
        #             file.write('\n')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train inversion model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--root_path', type=str, default='MaskNet/hypernet/model_1/sensitivities', help='root path')

    args = parser.parse_args()
 
    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")
