import torch
import torch.nn as nn
import torch.nn.functional as F

class config(object):
    def __init__(self):
        # 数据相关配置
        self.data_path = './dataset'
        self.mean = 0.5                     # 数据正则化的均值
        self.std = 0.5                      # 数据正则化的方差

        # 训练相关配置
        self.batch_size = 512
        self.class_num = 10                 # 输出类别
        self.epoch = 200                    # 训练轮数
        self.learning_rate = 1e-5           # 学习率
        self.log_path = '../Logs/'          # 日志路径
        self.maxiter_without_improvement = 1000  # 若1000轮没有优化则退出
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用cuda训练








class AlexNet(nn.Module):

    def __init__(self, config):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 输入 (3,227,227)
            # 输出 (48,55,55)      (227-11+0)/4+1=55
            nn.Conv2d(in_channels=3,out_channels=48, kernel_size=(11,11), stride=4, padding=0),
            # 输出 (48,55,55)
            nn.ReLU(inplace=True),
            # 输出 (48,27,27)      (55-3+0)/2+1=27
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 输出 (128,27,27)     (27-5+4)/1+1=27
            nn.Conv2d(in_channels=48,out_channels=128,kernel_size=(5,5),stride=1,padding=2),
            nn.ReLU(inplace=True),
            # 输出 (128,13,13)      (27-3+0)/2+1=13
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 输出 (192,13,13)      (13-3+2)/1+1=13
            nn.Conv2d(in_channels=128,out_channels=192,kernel_size=(3, 3),stride=1,padding=1),
            # 输出 (192,13,13)
            nn.ReLU(inplace=True),
            # 输出 (192,13,13)      (13-3+2)/1+1=13
            nn.Conv2d(in_channels=192,out_channels=192,kernel_size=(3,3),stride=1,padding=1),
            # 输出 (192,13,13)
            nn.ReLU(inplace=True),
            # 输出 (128,13,13)       (13-3+2)/1+1=13
            nn.Conv2d(in_channels=192,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(inplace=True),
            # 输出 (128,6,6)         (13-3+0)/2+1=6
            nn.MaxPool2d(kernel_size=3, stride=2),
        )


        self.classifier = nn.ModuleList(
            nn.Linear(in_features=128*6*6, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=config.class_num),
        )





    def forward(self, x):
        x = self.features(x)

        x = self.classifier(x)

        return x
