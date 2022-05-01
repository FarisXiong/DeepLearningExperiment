import torch
import torch.nn as nn
import torch.nn.functional as F

class config(object):
    def __init__(self,model_name):
        # 数据相关配置
        self.experiment_path = './Experiment2/'                                     # 实验路径
        self.data_path = self.experiment_path + 'Caltech101'                        # 数据路径
        self.mean1 = 0.485                                                          # 数据正则化的均值
        self.mean2 = 0.456
        self.mean3 = 0.406
        self.std1 = 0.229                                                           # 数据正则化的方差
        self.std2 = 0.224 
        self.std3 = 0.225
        self.image_size = 224                                                       # 数据resize大小

        # 训练相关配置
        self.model_name = model_name
        self.batch_size = 64
        self.class_num = 101                                                        # 输出类别
        self.epoch = 500                                                            # 训练轮数
        self.learning_rate = 1e-5                                                   # 学习率
        self.log_path = self.experiment_path + 'Logs/'                              # 日志路径
        self.maxiter_without_improvement = 1000                                     # 若1000轮没有优化则退出
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用cuda训练
        self.save_model_path = self.experiment_path + 'Models/' + self.model_name + '.ckpt'








class AlexNet(nn.Module):

    def __init__(self, config, init_weights=True):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 输入 (3,224,224)
            # 输出 (96,55,55)      (224-11+4)/4+1=55
            nn.Conv2d(in_channels=3,out_channels=96, kernel_size=(11,11), stride=4, padding=2),
            # 输出 (96,55,55)
            nn.ReLU(inplace=True),
            # 输出 (96,27,27)      (55-3+0)/2+1=27
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 输出 (256,27,27)     (27-5+4)/1+1=27
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=(5,5),stride=1,padding=2),
            nn.ReLU(inplace=True),
            # 输出 (256,13,13)      (27-3+0)/2+1=13
            nn.MaxPool2d(kernel_size=(3, 3), stride=2),
            # 输出 (384,13,13)      (13-3+2)/1+1=13
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=(3, 3),stride=1,padding=1),
            # 输出 (384,13,13)
            nn.ReLU(inplace=True),
            # 输出 (384,13,13)      (13-3+2)/1+1=13
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=(3,3),stride=1,padding=1),
            # 输出 (384,13,13)
            nn.ReLU(inplace=True),
            # 输出 (256,13,13)       (13-3+2)/1+1=13
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=(3,3),stride=1,padding=1),
            nn.ReLU(inplace=True),
            # 输出 (256,6,6)         (13-3+0)/2+1=6
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.Flatten = nn.Flatten()
        self.Linear1 = nn.Sequential(
            nn.Linear(in_features=2*128*6*6, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.Linear2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
        )
        self.Linear3 = nn.Linear(in_features=4096,out_features=config.class_num)

        if init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x1 = self.features(x)
        x2 = self.Flatten(x1)
        x3 = self.Linear1(x2)
        x4 = self.Linear2(x3)
        x5 = self.Linear3(x4)
        return x5




class VGG(nn.Module):

    def __init__(self, config, init_weights=True):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            # 输入 (3,224,224)
            # 输出 (64,224,224)
            # (224-3+2)/1+1=224
            # 224
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            # 输入 (64,224,224)
            # 输出 (128,112,112)
            # (224-2)/2+1=112
            # 112
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
            # (112-3+2)/1+1=112
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            
            # 56
            # (112-2)/2+1=56
            # (56-3+2)/1+1=56
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),

            # 28
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),

            # 14
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            nn.ReLU(),

            # 输出 (512*7*7)
            nn.MaxPool2d(kernel_size=(2,2), stride=2, padding=0),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=config.class_num)
        )
        if init_weights:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        outputsFeatures = self.features(x)
        outputsFlatten = self.flatten(outputsFeatures)
        outputs = self.classifier(outputsFlatten)
        return outputs







