import pandas as pd
import nltk
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

def getJoinTitle(train_path, test_path, titles_path):
    """
    将数据与titles表连接
    :return: 将表连接后的数据
    """
    train = pd.read_csv(train_path)
    titles = pd.read_csv(titles_path, dtype=str)
    merged_train = train.merge(titles, left_on='context', right_on='code')
    merged_train['input'] = merged_train['anchor'] + '[SEP]' + merged_train['title']
    merged_train['target'] = merged_train['target'] + '[SEP]' + merged_train['title']
    merged_train['all'] = merged_train['anchor'] + '[SEP]' + merged_train['target'] + '[SEP]' + merged_train['title']
    return merged_train


def fenci(x):
    # print(x['all'])
    x = tokenizer(x['all'])
    lengthList.append(len(x['input_ids']))
    return len(x['input_ids'])

input_path = './input/'
train_path = input_path + 'train.csv'
test_path = input_path + 'test.csv'
titles_path = input_path + 'titles.csv'
trainData = getJoinTitle(train_path, test_path, titles_path)

tokenizer = AutoTokenizer.from_pretrained('./pretrained')
lengthList = []
trainData['all_length'] = trainData.apply(fenci, axis=1)

# print(range(len(lengthList)))
plt.xlabel("x")
plt.ylabel("length")
plt.bar([i for i in range(len(lengthList))],lengthList,width=0.8,)
plt.ylim(0,250)
plt.savefig('./a.png')
plt.show()
