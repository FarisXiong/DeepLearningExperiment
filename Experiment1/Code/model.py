import torch
import torch.nn as nn
import torch.nn.functional as F


class config(object):
    def __init__(self):
        # 数据相关配置
        self.data_path = './dataset'
        self.mean = 0.5  # 数据正则化的均值
        self.std = 0.5  # 数据正则化的方差


        # 训练相关配置
        self.batch_size = 512
        self.class_num = 10  # 输出类别
        self.epoch = 200     # 训练轮数
        self.learning_rate = 1e-5  # 学习率
        self.log_path = '../Logs/'  # 日志路径
        self.maxiter_without_improvement = 1000  # 若1000轮没有优化则退出
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用cuda训练


class mlpConfig(config):
    def __init__(self):
        super(mlpConfig, self).__init__()
        # MLP相关配置
        self.Linear1 = 28 * 28
        self.Linear2 = 512
        self.Linear3 = 512
        self.save_model_path = '../Models/MLP.ckpt'  # 模型保存路径


class cnnConfig(config):
    def __init__(self):
        super(cnnConfig, self).__init__()
        # CNN网络相关配置
        self.out_channels = 64  # 卷积输出通道数
        self.filter_size = 4  # 定义卷积核的size
        self.hidden_size = 1024  # 隐藏层大小
        self.save_model_path = '../Models/CNN1.ckpt'  # 模型保存路径


class CNN(nn.Module):

    def __init__(self, cnnConfig):
        super(CNN, self).__init__()
        self.Conv = nn.Conv2d(1, cnnConfig.out_channels, (cnnConfig.filter_size, cnnConfig.filter_size))
        self.Maxpool = nn.MaxPool2d(2)
        self.Flatten = nn.Flatten()
        self.Fc1 = nn.Linear(
            (int((28 - cnnConfig.filter_size + 1) / 2)) * (int((28 - cnnConfig.filter_size + 1) / 2)) * 64,
            cnnConfig.hidden_size)
        self.Fc2 = nn.Linear(cnnConfig.hidden_size, cnnConfig.class_num)
        self.Relu = nn.ReLU()

    def forward(self, x):
        output = self.Conv(x)
        output = self.Maxpool(output)
        output = self.Flatten(output)
        output = self.Relu(output)
        output = self.Fc1(output)
        output = self.Fc2(output)
        return output


class MLP(nn.Module):
    def __init__(self, mlpConfig):
        super(MLP, self).__init__()
        self.Flatten = nn.Flatten()
        self.Fc1 = nn.Linear(mlpConfig.Linear1, mlpConfig.Linear2)
        self.Fc2 = nn.Linear(mlpConfig.Linear2, mlpConfig.Linear3)
        self.Fc3 = nn.Linear(mlpConfig.Linear3, mlpConfig.class_num)
        self.Relu = nn.ReLU()

    def forward(self, x):
        output = self.Flatten(x)
        output = self.Fc1(output)
        output = self.Relu(output)
        output = self.Fc2(output)
        output = self.Relu(output)
        output = self.Fc3(output)
        return output
