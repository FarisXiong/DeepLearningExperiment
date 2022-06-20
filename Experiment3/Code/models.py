import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from transformers import AutoFeatureExtractor, SwinForImageClassification


class VGG(nn.Module):
    def __init__(self, config:config, init_weights=True):
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


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.basicblock(input)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class BottleNeck(nn.Module):
    # 维度扩张
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.bottleneck(input)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    """
    ResNet的具体实现
    """
    def __init__(self, block, num_layer, n_classes, init_weights=True):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # 定义网络结构
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, num_layer[0])
        self.layer2 = self._make_layer(block, 128, num_layer[1], 2)
        self.layer3 = self._make_layer(block, 256, num_layer[2], 2)
        self.layer4 = self._make_layer(block, 512, num_layer[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, n_classes)
        if init_weights == True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)



    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x




def ResNet18(n_classes: int):
    """
    构造ResNet18模型
    :return: ResNet18
    """
    model = ResNet(block=BasicBlock, num_layer=[2, 2, 2, 2], n_classes=n_classes)
    return model

def ResNet34(n_classes: int):
    """
    构造ResNet34模型
    :return: ResNet34
    """
    model = ResNet(block=BasicBlock, num_layer=[3, 4, 6, 3], n_classes=n_classes)
    return model


def ResNet50(n_classes: int):
    """
    构造ResNet50模型
    :return: ResNet50
    """
    model = ResNet(block=BottleNeck, num_layer=[3, 4, 6, 3], n_classes=n_classes)
    return model

def ResNet101(n_classes: int):
    """
    构造ResNet101模型
    :return: ResNet101
    """
    model = ResNet(block=BottleNeck, num_layer=[3, 4, 23, 3], n_classes=n_classes)
    return model

def ResNet152(n_classes: int):
    """
    构造ResNet152模型
    :return: ResNet152
    """
    model = ResNet(block=BottleNeck, num_layer=[3, 8, 36, 3], n_classes=n_classes)
    return model







class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(SEBasicBlock, self).__init__()
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        if out_channels == 64:
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)
        elif out_channels == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif out_channels == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif out_channels == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.seblock = nn.Sequential(
            self.globalAvgPool,
            nn.Flatten(),
            nn.Linear(in_features=out_channels, out_features=round(out_channels/16)),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=round(out_channels/16), out_features=out_channels),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.basicblock(input)
        if self.downsample:
            residual = self.downsample(residual)

        origin_x = x
        x = self.seblock(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1) * origin_x

        x += residual
        x = self.relu(x)
        return x

class SEBottleNeck(nn.Module):
    # 维度扩张
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(SEBottleNeck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        if out_channels == 64:
            self.globalAvgPool = nn.AvgPool2d(56, stride=1)
        elif out_channels == 128:
            self.globalAvgPool = nn.AvgPool2d(28, stride=1)
        elif out_channels == 256:
            self.globalAvgPool = nn.AvgPool2d(14, stride=1)
        elif out_channels == 512:
            self.globalAvgPool = nn.AvgPool2d(7, stride=1)
        self.seblock = nn.Sequential(
            self.globalAvgPool,
            nn.Flatten(),
            nn.Linear(in_features=out_channels, out_features=round(out_channels/16)),
            nn.ReLU(inplace=False),
            nn.Linear(in_features=round(out_channels/16), out_features=out_channels),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, input):
        residual = input
        x = self.bottleneck(input)
        if self.downsample:
            residual = self.downsample(residual)

        origin_x = x
        x = self.seblock(x)
        x = x.view(x.shape[0], x.shape[1], 1, 1) * origin_x

        x += residual
        x = self.relu(x)
        return x

class SENet(nn.Module):
    """
    定义SENet模型
    """
    def __init__(self, block, num_layer, n_classes, init_weights=True):
        super(SENet, self).__init__()
        self.in_channels = 64
        # 定义网络结构
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.layer1 = self._make_layer(block, 64, num_layer[0])
        self.layer2 = self._make_layer(block, 128, num_layer[1], 2)
        self.layer3 = self._make_layer(block, 256, num_layer[2], 2)
        self.layer4 = self._make_layer(block, 512, num_layer[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * block.expansion, n_classes)
        if init_weights == True:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = F.softmax(x, dim=1)
        return x

def SENet18(n_classes: int):
    """
    构造SENet18模型
    :return: SENet18
    """
    model = SENet(block=SEBasicBlock, num_layer=[2, 2, 2, 2], n_classes=n_classes)
    return model

def SENet34(n_classes: int):
    """
    构造SENet34模型
    :return: SENet34
    """
    model = SENet(block=SEBasicBlock, num_layer=[3, 4, 6, 3], n_classes=n_classes)
    return model


def SENet50(n_classes: int):
    """
    构造SENet50模型
    :return: SENet50
    """
    model = SENet(block=SEBottleNeck, num_layer=[3, 4, 6, 3], n_classes=n_classes)
    return model

def SENet101(n_classes: int):
    """
    构造SENet101模型
    :return: SENet101
    """
    model = SENet(block=SEBottleNeck, num_layer=[3, 4, 23, 3], n_classes=n_classes)
    return model

def SENet152(n_classes: int):
    """
    构造SENet152模型
    :return: SENet152
    """
    model = SENet(block=SEBottleNeck, num_layer=[3, 8, 36, 3], n_classes=n_classes)
    return model



class SwinTransformer(nn.Module):
    def __init__(self, n_classes):
        super(SwinTransformer, self).__init__()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        self.fc = nn.Linear(1000, n_classes)

    def forward(self, inputs):
        for tensor in inputs.cpu():
            tensor_pro = self.feature_extractor(tensor, return_tensors="pt")

        logits = self.model(inputs).logits
        outputs = self.fc(logits)
        outputs = F.softmax(outputs, dim=1)
        return outputs
