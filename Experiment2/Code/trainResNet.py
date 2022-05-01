import torchvision
from model import config
from getdata import dataList
from getdata import myDataset
from torch.utils.data.dataloader import DataLoader
from train_eval import train
from train_eval import test
import torch.nn as nn

if __name__ == '__main__':
    myConfig = config('ResNet50')
    DataList = dataList(myConfig)
    # 加载数据
    trainDataSet = myDataset(DataList.train, myConfig)
    devDataSet = myDataset(DataList.dev, myConfig)
    testDataSet = myDataset(DataList.test, myConfig)
    trainDataLoader = DataLoader(dataset=trainDataSet, batch_size=myConfig.batch_size, shuffle=True)
    devDataLoader = DataLoader(dataset=devDataSet, batch_size=myConfig.batch_size, shuffle=False)
    testDataLoader = DataLoader(dataset=testDataSet, batch_size=myConfig.batch_size, shuffle=False)
    ResNet = torchvision.models.resnet.resnet50(pretrained=True)
    ResNet.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=myConfig.class_num, bias=True),
    )
    ResNet.to(myConfig.device)
    train(myConfig, ResNet, trainDataLoader, devDataLoader)
    ResNet = torchvision.models.resnet.resnet50(pretrained=True)
    ResNet.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=myConfig.class_num, bias=True),
    )
    test(myConfig, ResNet, testDataLoader)
    














