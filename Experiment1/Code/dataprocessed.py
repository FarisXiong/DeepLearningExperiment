import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy

if __name__ == "__main__":

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
    path = './dataset'  # 数据集下载后保存的目录

    # 下载训练集和测试集
    trainData = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
    testData = torchvision.datasets.MNIST(path, train=False, transform=transform)






