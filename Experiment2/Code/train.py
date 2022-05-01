from model import AlexNet
from model import config
from getdata import dataList
from getdata import myDataset
from torch.utils.data.dataloader import DataLoader
from train_eval import train
from train_eval import test

if __name__ == '__main__':
    myConfig = config('AlexNet')
    DataList = dataList(myConfig)
    # 加载数据
    trainDataSet = myDataset(DataList.train, myConfig)
    devDataSet = myDataset(DataList.dev, myConfig)
    testDataSet = myDataset(DataList.test, myConfig)
    trainDataLoader = DataLoader(dataset=trainDataSet, batch_size=myConfig.batch_size, shuffle=True)
    devDataLoader = DataLoader(dataset=devDataSet, batch_size=myConfig.batch_size, shuffle=False)
    testDataLoader = DataLoader(dataset=testDataSet, batch_size=myConfig.batch_size, shuffle=False)
    alexNet = AlexNet(myConfig).to(myConfig.device)
    train(myConfig, alexNet, trainDataLoader, devDataLoader)
    alexNet = AlexNet(myConfig)
    test(myConfig, alexNet, testDataLoader)
    














