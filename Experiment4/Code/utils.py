from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from config import config
from sklearn.model_selection import train_test_split
from dataset import textDataset
from dataset import weatherDataset
import pandas as pd
import jieba
import time
import datetime
import numpy as np

# def encodePreprocess(row):
#     # print(row['review'])
#     row = row['review'].encode('utf-8').decode('utf-8')
#     return row




def getData(config):
    """
    获取数据
    :return: 训练集、测试集、验证集的DataLoader
    """

    if config.data_type == 'Text':
        Data = pd.read_csv(config.input_path)
        textData = Data.dropna(subset=['review'])
        # textData['review_proc'] = textData.apply(encodePreprocess, axis=1)
        textTrain = textData.iloc[[i for i in range(len(textData)) if i % 5 == 1 or i % 5 == 2 or i % 5 == 3]]
        textDev = textData.iloc[[i for i in range(len(textData)) if i % 5 == 4]]
        textTest = textData.iloc[[i for i in range(len(textData)) if i % 5 == 0]]
        trainDataset = textDataset(textTrain, config)
        devDataset = textDataset(textDev, config)
        testDataset = textDataset(textTest, config)
        trainDataLoader = DataLoader(dataset=trainDataset, batch_size=config.batch_size, shuffle=True)
        devDataLoader = DataLoader(dataset=devDataset, batch_size=config.batch_size, shuffle=True)
        testDataLoader = DataLoader(dataset=testDataset, batch_size=config.batch_size, shuffle=False)
        return (trainDataLoader, devDataLoader, testDataLoader)

    else:

        trainData = pd.read_csv(config.train_path)
        testData = pd.read_csv(config.test_path)
        trainData = [data[1] for data in list(trainData.groupby('WeekIndex')) if len(data[1]) == 1008]
        testData = [data[1] for data in list(testData.groupby('WeekIndex')) if len(data[1]) == 1008]
        trainDataset = weatherDataset(trainData,)
        testDataset = weatherDataset(testData, )
        trainDataLoader = DataLoader(dataset=trainDataset, batch_size=config.batch_size, shuffle=True)
        testDataLoader = DataLoader(dataset=testDataset, batch_size=config.batch_size, shuffle=False)
        return (trainDataLoader, testDataLoader)

