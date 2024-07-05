import os
import numpy as np
import torch
from scipy.special import kl_div
from torch.utils.data import random_split
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import feature_model_med_1
import argparse

def calculate_kl_divergence(models_dir):
    transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    dataset_path = 'feature_extractor_dataset'
    full_dataset = datasets.ImageFolder(dataset_path, transform=transform)

    # Define the proportions for training and test sets
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Split the dataset
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # 获取所有.pth模型的路径
    model_files = [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith('.pth')]
    
    probs = []
    for model_file in model_files:
        # 加载模型
        model = feature_model_med_1.FaceRecognitionModel(num_classes=20)
        model_dict = torch.load(model_file, map_location=torch.device('cpu'))
        model.load_state_dict(model_dict)  # 加载模型参数
        model.eval()  # 设置模型为评估模式
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                predictions = model(inputs)  # Assuming model expects inputs as the first argument
                probs.append(predictions.numpy())

    max_samples = max(len(prob) for prob in probs)
    probs = [np.pad(prob, ((0, max_samples - len(prob)), (0, 0)), mode='constant', constant_values=0) for prob in probs]

    # 确保概率分布是有效的，即在 [0, 1] 范围内并且每行的和为1
    probs = [np.clip(prob, 1e-10, 1) for prob in probs]

    # 初始化KL散度矩阵
    num_models = len(probs)
    kl_matrix = np.zeros((num_models, num_models))

    # 计算每对模型之间的KL散度
    for i in range(num_models):
        for j in range(num_models):
            if i != j:
                kl_divergence = np.sum(probs[i] * np.log(probs[i] / probs[j]), axis=1)
                kl_matrix[i, j] = np.mean(kl_divergence)

    return np.mean(kl_matrix)

def main():
    parser = argparse.ArgumentParser(description='Calculate KL Divergence between models.')
    parser.add_argument('--models_dir', type=str, required=True, help='Directory containing the model files.')

    args = parser.parse_args()
    models_dir = args.models_dir

    kl_divergences = [calculate_kl_divergence(models_dir) for _ in range(5)]

    print("Average KL Divergences over 5 runs:")
    print(" ".join(map(str, kl_divergences)))

if __name__ == "__main__":
    main()
