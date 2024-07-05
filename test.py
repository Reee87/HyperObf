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

    # Create a dataset using ImageFolder
    train_dataset = datasets.ImageFolder(train_dataset_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_dataset_path, transform=transform)

    fixed_label = 50
    res = []
    for item in os.listdir(args.root_path):
        target_test = "users/test"
        target_train = os.path.join(args.root_path, item)
        target_test = os.path.join(target_test, item)
        target_test_dataset = datasets.ImageFolder(target_test, transform=transform,target_transform= lambda x: fixed_label)
        target_train_dataset = datasets.ImageFolder(target_train, transform=transform,target_transform= lambda x: fixed_label)

        combined_train_dataset = ConcatDataset([train_dataset,target_train_dataset])
        combined_test_dataset = ConcatDataset([test_dataset,target_test_dataset])

        new_train_classes = list(set(train_dataset.classes) |set(target_train_dataset.classes)) 
        new_test_classes = list(set(test_dataset.classes) | set(target_test_dataset.classes))

        # combined_train_dataset.classes = new_train_classes
        # combined_test_dataset.classes = new_test_classes
        num_classes = len(new_test_classes)

        train_loader = DataLoader(combined_train_dataset, batch_size=16, shuffle=True, num_workers=4)
        test_loader = DataLoader(combined_test_dataset, batch_size=16, shuffle=True, num_workers=4)

        # Define a simple face recognition model (you can customize the architecture)

        # Instantiate the model and move it to the device
        model = FaceRecognitionModel(num_classes=num_classes).to(device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=0.0001) 

        # Training loop
        num_epochs = 30
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
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
                for inputs, labels in train_loader:
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
                for inputs, labels in test_loader:
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

            # Print training and test accuracy
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}")
            print(f"Accuracy for Class {class_of_interest}: {accuracy_class}")

            if epoch == 29:
                res.append(accuracy_class)
        # Instantiate the model
        
        # conf_matrix = confusion_matrix(all_labels, all_predicted)
        # class_report = classification_report(all_labels, all_predicted, target_names=test_dataset.classes)

        # print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}")
        # print("Confusion Matrix:")
        # print(conf_matrix)
        # conf_matrix_str = str(confusion_matrix.numpy())

        # Write the string to a file
        # with open('matrix.txt', 'w') as f:
        #     f.write(conf_matrix_str)
        # print("Classification Report:")
        # print(class_report)
        
        
        # model_path = 'test_model_resnet18.pth'
        # torch.save(model.state_dict(), model_path)

        model.train()  # Set the model back to training mode for the next epoch
    
    output_dir = f"output/{args.root_path}"
    os.makedirs(output_dir, exist_ok=True)
    with open(f"output/{args.root_path}/output.txt", "w") as file:
        for item in res:
            file.write("%s\n" % item)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train inversion model.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--root_path', type=str, default=None, help='root path')

    args = parser.parse_args()
 
    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")