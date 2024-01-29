"""
  MNIST 
  Zhihao Zhan(zhanzhihao_dt@163.com)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

from unet import UNet


class add_gaussian_noise():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


def main():
    std_added_noise = 20
    data_transform = {
        'transformation_adding_noise': transforms.Compose([
            transforms.ToTensor(),
            add_gaussian_noise(std=std_added_noise/255),
            transforms.Resize(64),
            transforms.ConvertImageDtype(torch.float)
        ]),
        'transformation_original': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(64),
            transforms.ConvertImageDtype(torch.float)
        ])
    }

    mnist_trainset = datasets.MNIST(
        root='./data', train=True, download=True, transform=data_transform['transformation_original'])
    mnist_trainset_noisy = datasets.MNIST(
        root='./data', train=True, download=True, transform=data_transform['transformation_adding_noise'])

    mnist_trainset_input = []
    for input_tensor, _ in mnist_trainset_noisy:
        mnist_trainset_input.append(input_tensor)

    mnist_trainset_output = []
    for input_tensor, _ in mnist_trainset:
        mnist_trainset_output.append(input_tensor)

    train_loader_input = DataLoader(
        mnist_trainset_input,
        batch_size=128,
        num_workers=2,
        shuffle=False)
    train_loader_output = DataLoader(
        mnist_trainset_output,
        batch_size=128,
        num_workers=2,
        shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    training_loss = []
    for epoch in range(5):
        i = 0
        for input_data, output_data in zip(train_loader_input, train_loader_output):
            input_data = input_data.to(device)
            output_data = output_data.to(device)

            optimizer.zero_grad()

            outputs = model(input_data)
            loss = criterion(outputs, output_data)
            loss.backward()
            optimizer.step()

            if i % 100 == 99:
                print(
                    f'#Epoch: {epoch + 1}, #Batch {i + 1:5d}, Training Loss: {loss.item():.6f}')
                training_loss.append(loss.item())
            i += 1

    plt.plot(training_loss)
    plt.title('Learning Curve')
    plt.xlabel('#iteration')
    plt.ylabel('Training Loss')
    plt.show()

    mnist_testset = datasets.MNIST(
        root='./data', train=False, download=True, transform=data_transform['transformation_original'])
    mnist_testset_noisy = datasets.MNIST(
        root='./data', train=False, download=True, transform=data_transform['transformation_adding_noise'])

    sample_number = 100
    original_image = mnist_testset[sample_number][0][0]
    noisy_image = mnist_testset_noisy[sample_number][0][0]
    output_denoised = model(mnist_testset_noisy[sample_number][0].unsqueeze(
        dim=0).to(device)).cpu().detach().numpy().reshape((64, 64))

    f, axes = plt.subplots(1, 3)
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(noisy_image, cmap='gray')
    axes[1].set_title('Noisy Image')
    axes[1].axis('off')
    axes[2].imshow(output_denoised, cmap='gray')
    axes[2].set_title('Network Output')
    axes[2].axis('off')
    plt.show()


if __name__ == "__main__":
    main()
