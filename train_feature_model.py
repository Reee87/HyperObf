import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix, classification_report
import feature_model_med_1
from torch.utils.data import random_split
import os
import matplotlib.pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    augmentation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Adjust color
    transforms.RandomRotation(degrees=10),  # Randomly rotate the image up to 10 degrees
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    # Compose the transformations for both resizing and augmentation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
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
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    # Instantiate the model and move it to the device
    model = feature_model_med_1.FaceRecognitionModel(num_classes=num_classes).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Training loop
    num_epochs = 500
    train_accuracies = []
    test_accuracies = []

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

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())

        accuracy_test = correct_test / total_test

        # Calculate and print confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predicted)
        class_report = classification_report(all_labels, all_predicted, target_names=full_dataset.classes)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Train Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)
        
        train_accuracies.append(accuracy_train)
        test_accuracies.append(accuracy_test)

        model_path = 'MaskNet/scratch/model.pth'
        torch.save(model.state_dict(), model_path)

        model.train()

if __name__ == "__main__":
    main()
