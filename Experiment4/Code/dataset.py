from torch.utils.data import Dataset
import jieba
from config import config
import torch
import logging
jieba.setLogLevel(logging.INFO)

labelMap = {
    '书籍': 0,
    '平板': 1,
    '手机': 2,
    '水果': 3,
    '洗发水': 4,
    '热水器': 5,
    '蒙牛': 6,
    '衣服': 7,
    '计算机': 8,
    '酒店': 9,
}

class textDataset(Dataset):
    def __init__(self, df, config:config):
        super(textDataset, self).__init__()
        self.df = df
        self.index_to_key = config.index_to_key
        self.key_to_index = config.key_to_index
        self.padding_size = config.padding_size

    def __getitem__(self, item):
        Item = self.df.iloc[item]
        text = Item['review']
        wordList = [word for word in jieba.cut(text, cut_all=False)]
        vectors = [self.key_to_index[word] for word in wordList]
        for word in wordList:
            if word in self.index_to_key:
                vectors.append(self.key_to_index[word])
            else:
                vectors.append(self.key_to_index['[UNK]'])
        if len(vectors) < self.padding_size:
            vectors.extend([self.key_to_index['[PAD]'] for i in range(self.padding_size - len(vectors))])
        else:
            vectors = vectors[0:self.padding_size]
        return {
            'text': torch.LongTensor(vectors),
            'label': labelMap[Item['cat']]
        }

    def __len__(self):
        return len(self.df)

class weatherDataset(Dataset):
    def __init__(self, dflist):
        super(weatherDataset, self).__init__()
        self.dflist = dflist

    def __getitem__(self, item):
        # self.df
        df = self.dflist[item]
        trainData = df.loc[df['WeekDay'] <= 4]
        resultData = df.loc[df['WeekDay'] >= 5]

        date = list(resultData['Date Time'])
        trainData = trainData.drop(['Date Time'], axis=1)
        resultData = resultData['T (degC)']

        # 标准化
        # trainData = (trainData-trainData.mean())/trainData.std()
        trainData = trainData.values
        resultData = resultData.values

        return {
            'train': torch.Tensor(trainData),
            'result': torch.Tensor(resultData),
            'date': date,
        }


    def __len__(self):
        return len(self.dflist)





