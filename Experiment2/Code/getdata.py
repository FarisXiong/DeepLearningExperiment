import torchvision
import torch

if __name__ == "__main__":

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
    path = './dataset'  # 数据集下载后保存的目录

    # 下载训练集和测试集
    data = torchvision.datasets.Caltech101(path, download=True)





