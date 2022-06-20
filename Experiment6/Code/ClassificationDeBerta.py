from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

from transformers import DebertaTokenizer


class config():
    """
    模型相关配置
    """
    def __init__(self):
        # 数据相关路径
        self.input_path = './input/'
        self.train_path = self.input_path + 'train.csv'
        self.test_path = self.input_path + 'test.csv'
        self.titles_path = self.input_path + 'titles.csv'
        self.kFoldNum = 5
        self.kFoldList = ['./'+str(i)+'Fold/' for i in range(0, self.kFoldNum)]

        # 模型相关数据
        self.model_path = './pretrained'
        self.tokenizer_path = './pretrained'

        # 训练相关设置
        self.learning_rate = 1e-6   # 学习率
        self.epoch = 5              # 训练轮数
        self.batch_size = 16        # batch_size



def getJoinTitle(config, path):
    """
    将数据与titles表连接
    :return: 将表连接后的数据
    """
    train = pd.read_csv(path)
    titles = pd.read_csv(config.titles_path, dtype=str)
    merged_train = train.merge(titles, left_on='context', right_on='code')
    merged_train['input'] = merged_train['anchor'] + '</s>' + merged_train['title']
    merged_train['target'] = merged_train['target'] + '</s>' + merged_train['title']
    return merged_train


class myDataSet(Dataset):
    """
    对Dataset的继承
    """
    def __init__(self,dataframe):
        self.input = dataframe['input']
        self.target = dataframe['target']
        if 'score' not in dataframe.columns:
            self.has_score = False
            self.score = None
        else:
            self.has_score = True
            self.score = dataframe['score']

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        inputs = self.input.iloc[item]
        targets = self.target.iloc[item]
        if self.has_score == False:
            return {
                **tokenizer(inputs, targets)
            }
        else:
            return {
                **tokenizer(inputs, targets),
                'label': float(self.score.iloc[item])
            }



def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.reshape(len(predictions))
    return {
        'pearson': np.corrcoef(predictions, labels)[0][1]
    }


if __name__ == '__main__':
    configModel = config()
    # 获取训练数据
    trainData = getJoinTitle(configModel, configModel.train_path)
    kfold = KFold(n_splits=configModel.kFoldNum,shuffle=True)
    foldIndex = 0
    model = AutoModelForSequenceClassification.from_pretrained(configModel.model_path, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(configModel.tokenizer_path)
    for train_index, eval_index in kfold.split(trainData):

        trainDataset = myDataSet(trainData.iloc[train_index])
        evalDataset = myDataSet(trainData.iloc[eval_index])
        train_args = TrainingArguments(
            output_dir=configModel.kFoldList[foldIndex],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=configModel.learning_rate,
            per_device_train_batch_size=configModel.batch_size,
            per_device_eval_batch_size=configModel.batch_size,
            num_train_epochs=configModel.epoch,
            load_best_model_at_end=True,
            metric_for_best_model="pearson",
            save_total_limit=1
        )
        foldIndex += 1
        trainer = Trainer(
            model,
            train_args,
            train_dataset=trainDataset,
            eval_dataset=evalDataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer.evaluate()
        trainer.train()


