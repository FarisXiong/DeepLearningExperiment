from model import MLP
from model import mlpConfig
from model import CNN
from model import cnnConfig
from model import config
import torch
import torch.utils.data.dataset
import torchvision
from train_eval import train
from torch.utils import data

if __name__ == '__main__':
    # 加载配置信息
    myConfig = cnnConfig()
    # 加载数据集
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(myConfig.mean, myConfig.std)])
    trainData = torchvision.datasets.MNIST(myConfig.data_path, transform=transform, train=True, download=True)
    testData = torchvision.datasets.MNIST(myConfig.data_path, transform=transform, train=False)

    train_iter = data.DataLoader(trainData, batch_size=myConfig.batch_size, shuffle=True)
    test_iter = data.DataLoader(testData, batch_size=myConfig.batch_size, shuffle=True)

    # 训练模型
    model = CNN(myConfig).to(myConfig.device)
    train(myConfig, model, train_iter, test_iter)

    # model = MLP(myConfig).to(myConfig.device)
    # train(myConfig, model, train_iter, test_iter)

