from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np



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
        self.kFoldList = ['./' + str(i) + 'Fold/checkpoint-3648/' for i in range(0, self.kFoldNum)]
        self.kFoldList.extend(['./' + str(i) + 'Fold/checkpoint-4560/' for i in range(0, self.kFoldNum)])

        # 模型相关数据
        self.model_path = './pretrained'
        self.tokenizer_path = './pretrained'

        # 训练相关设置
        self.learning_rate = 1e-6   # 学习率
        self.epoch = 5              # 训练轮数
        self.batch_size = 32        # batch_size



def getJoinTitle(config, path):
    """
    将数据与titles表连接
    :return: 将表连接后的数据
    """
    train = pd.read_csv(path)
    titles = pd.read_csv(config.titles_path, dtype=str)
    merged_train = train.merge(titles, left_on='context', right_on='code')
    merged_train['input'] = merged_train['title'] + '</s>' + merged_train['anchor']
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
    tokenizer = AutoTokenizer.from_pretrained(configModel.tokenizer_path)

    testData = getJoinTitle(configModel, configModel.test_path)
    testDataset = myDataSet(testData)
    predictions = []
    for fold in range(configModel.kFoldNum*2):
        foldName = configModel.kFoldList[fold]
        model = AutoModelForSequenceClassification.from_pretrained(foldName, num_labels=1)
        trainer = Trainer(model,tokenizer=tokenizer)
        outputs = trainer.predict(testDataset)
        prediction = outputs.predictions
        predictions.append(prediction)
    score = np.mean(predictions,axis=0)
    predict_result = np.column_stack((testData['id'],score))
    df = pd.DataFrame(predict_result, columns=['id', 'score'])
    df.to_csv('./submission.csv',index=False)

