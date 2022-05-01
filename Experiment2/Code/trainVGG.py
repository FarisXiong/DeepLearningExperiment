from model import VGG
from model import config
from getdata import dataList
from getdata import myDataset
from torch.utils.data.dataloader import DataLoader
from train_eval import train
from train_eval import test
import torch

if __name__ == '__main__':
    myConfig = config('VGG16')
    DataList = dataList(myConfig)
    # 加载数据
    trainDataSet = myDataset(DataList.train, myConfig)
    devDataSet = myDataset(DataList.dev, myConfig)
    testDataSet = myDataset(DataList.test, myConfig)
    trainDataLoader = DataLoader(dataset=trainDataSet, batch_size=myConfig.batch_size, shuffle=True)
    devDataLoader = DataLoader(dataset=devDataSet, batch_size=myConfig.batch_size, shuffle=False)
    testDataLoader = DataLoader(dataset=testDataSet, batch_size=myConfig.batch_size, shuffle=False)
    VGG16 = VGG(myConfig).to(myConfig.device)
    train(myConfig, VGG16, trainDataLoader, devDataLoader)
    VGG16 = VGG(myConfig)
    test(myConfig, VGG16, testDataLoader)
    














