import torch
import time
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn import metrics
from utils import getData
from model import textRNN
from model import textGRU
from model import textLSTM
from model import textBiLSTM
from model import WeatherLSTM
from config import config
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
import functools

class trainer:
    def __init__(self, config:config):
        if config.data_type == 'Text':
            self.trainer_ = TextTrainer(config)
        else:
            self.trainer_ = WeatherTrainer(config)

    def train(self):
        self.trainer_.train()

    def test(self):
        self.trainer_.test()


class TextTrainer:
    def __init__(self, config:config):
        # 数据相关
        self.data_type = config.data_type
        self.DataLoaders = getData(config)
        # 模型相关
        self.model_name = config.model
        if config.model == 'RNN':
            self.model = textRNN(config)
        elif config.model == 'GRU':
            self.model = textGRU(config)
        elif config.model == 'LSTM':
            self.model = textLSTM(config)
        elif config.model == 'BiLSTM':
            self.model = textBiLSTM(config)
        # 训练相关
        self.device = config.device
        self.maxiter_without_improvement = config.maxiter_without_improvement
        self.model_saved = config.model_saved + self.data_type + '_' + self.model_name + '.pkl'
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.optimizer = config.optimizer
        self.log_path = config.log_path
        self.learning_rate = config.learning_rate
        self.images_saved = config.images_saved

    def train(self):
        start_time = time.time()
        # 设置优化器
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam((param for param in self.model.params), lr=self.learning_rate)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD((param for param in self.model.params), lr=self.learning_rate)

        total_batch = 0                # 记录总共训练的批次
        dev_best_loss = float('inf')   # 记录验证集上最低的loss
        dev_best_micro_f1score = float(0)    # 记录验证集上最高的acc
        dev_best_macro_f1score = float(0)    # 记录验证集上最高的f1score
        last_improve = 0               # 记录上一次dev的loss下降时的批次
        # flag = False                   # 是否结束训练

        writer = SummaryWriter(log_dir=self.log_path + self.data_type + "_" + self.model_name)
        for epoch in range(self.epoch):
            print("Epoch [{}/{}]".format(epoch+1, self.epoch))
            for trainsData in self.DataLoaders[0]:
                trains = trainsData['text'].to(self.device)
                labels = trainsData['label'].to(self.device)
                outputs = self.model(trains)
                # self.model.zero_grad()
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                if self.model_name == 'RNN' and self.data_type == 'Text':
                    self.model.grad_clipping()
                self.optimizer.step()
                # 输出当前效果
                if total_batch % 10 == 0:
                    ground_truth = labels.data.cpu()
                    predict_labels = torch.max(outputs.data,1)[1].cpu()
                    train_acc = metrics.accuracy_score(ground_truth, predict_labels)
                    dev_loss, dev_micro_f1_score, dev_macro_f1_score = self.eval()
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        self.model.save(self.model_saved)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    if dev_micro_f1_score > dev_best_micro_f1score:
                        dev_best_micro_f1score = dev_micro_f1_score 
                    if dev_macro_f1_score > dev_best_macro_f1score:
                        dev_best_macro_f1score = dev_macro_f1_score 
                    print("Iter:{:4d} TrainLoss:{:.12f} TrainAcc:{:.5f} DevLoss:{:.12f} DevMicroF1Score:{:.5f} DevMacroF1Score:{:.5f} Improve:{}".format(total_batch, loss.item(), train_acc, dev_loss, dev_micro_f1_score, dev_macro_f1_score, improve))
                    writer.add_scalar("loss/train", loss.item(), total_batch)
                    writer.add_scalar("loss/dev", dev_loss, total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                    writer.add_scalar("MicroF1Score/dev", dev_micro_f1_score, total_batch)
                    writer.add_scalar("MacroF1Score/dev", dev_macro_f1_score, total_batch)
                    # self.model.train()
                total_batch += 1
                if total_batch - last_improve > self.maxiter_without_improvement:
                    print("No optimization for a long time, modify learning rate..")
                    for params in self.optimizer.param_groups:  # 遍历Optimizer中的每一组参数
                        params['lr'] *= 0.9
                    # flag = True
                    # break
            # if flag:
                # break
        writer.close()
        end_time = time.time()
        print("Train Time : {:.3f} min , The Best Micro F1 Score in Dev : {} % , The Best Macro F1 Score in Dev : {}".format(((float)((end_time-start_time))/60), dev_best_micro_f1score,dev_best_macro_f1score))

    def eval(self):
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for testsData in self.DataLoaders[1]:
                tests = testsData['text'].to(self.device)
                labels = testsData['label'].to(self.device)
                outputs = self.model(tests)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                ground_truth = labels.cpu().data.numpy()
                predict_labels = torch.max(outputs.cpu().data, 1)[1].numpy()
                labels_all = np.append(labels_all, ground_truth)
                predict_all = np.append(predict_all, predict_labels)
        micro_f1_score = metrics.f1_score(labels_all, predict_all, average='micro')
        macro_f1_score = metrics.f1_score(labels_all, predict_all, average='macro')
        return loss_total / len(self.DataLoaders[1]), micro_f1_score, macro_f1_score

    def test(self):
        self.model.load(self.model_saved)
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for testsData in self.DataLoaders[1]:
                tests = testsData['text'].to(self.device)
                labels = testsData['label'].to(self.device)
                outputs = self.model(tests)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                ground_truth = labels.cpu().data.numpy()
                predict_labels = torch.max(outputs.cpu().data, 1)[1].numpy()
                labels_all = np.append(labels_all, ground_truth)
                predict_all = np.append(predict_all, predict_labels)
        columns = ['Book', 'Pad', 'Phone', 'Fruit', 'Shampoo', 'ElectricWaterHeaters', 'Monmilk', 'Clothe', 'Computer',
                   'Hotel']
        report = classification_report(labels_all, predict_all, target_names= columns)
        print("TEST:")
        print(report)
        # print(type(report))
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        df_cm = pd.DataFrame(confusion,
                             index=[i for i in columns],
                             columns=[i for i in columns])
        df_cm = df_cm.astype(int)
        sn.heatmap(df_cm,annot=True, fmt='.20g', cmap="BuPu")
        plt.savefig(self.images_saved +'confusion_'+self.data_type+'_'+self.model_name + '.png')



class WeatherTrainer:
    def __init__(self, config:config):
        # 数据相关
        self.data_type = config.data_type
        self.DataLoaders = getData(config)
        # 模型相关
        self.model_name = config.model
        self.model = WeatherLSTM(config).to(config.device)
        # 训练相关
        self.device = config.device
        self.maxiter_without_improvement = config.maxiter_without_improvement
        self.model_saved = config.model_saved + self.data_type + '_' + self.model_name + '.pkl'
        self.epoch = config.epoch
        self.batch_size = config.batch_size
        self.log_path = config.log_path
        self.learning_rate = config.learning_rate
        self.images_saved = config.images_saved
        self.output_result = config.output_result

    def train(self):
        start_time = time.time()
        # 设置优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=-1)
        total_batch = 0                     # 记录总共训练的批次
        best_r2_score = float('-inf')
        writer = SummaryWriter(log_dir=self.log_path + self.data_type + "_" + self.model_name)
        for epoch in range(self.epoch):
            print("Epoch [{}/{}]".format(epoch + 1, self.epoch))
            for trainsData in self.DataLoaders[0]:
                trains = trainsData['train'].to(self.device)
                results = trainsData['result'].to(self.device)
                outputs = self.model((trains, results))
                self.model.zero_grad()
                loss = F.l1_loss(outputs, results)
                loss.backward()
                optimizer.step()
                # 输出当前效果
                if total_batch % 4 == 0:
                    r2score = r2_score(results.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    if r2score > best_r2_score:
                        best_r2_score = r2score
                        torch.save(self.model.state_dict(), self.model_saved)
                    mse = mean_squared_error(results.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    mae = mean_absolute_error(results.cpu().detach().numpy(), outputs.cpu().detach().numpy())
                    print("Iter:{:4d} TrainLoss:{:.5f} TrainMseLoss:{:.5f} TrainMaeLoss:{:.5f} TrainR2Score:{:.4f}".format(total_batch, loss.item(), mse, mae, r2score))
                    writer.add_scalar("mseloss/train", mse, total_batch)
                    writer.add_scalar("maeloss/train", mae, total_batch)
                    writer.add_scalar("r2scoreloss/train", r2score, total_batch)
                    self.model.train()
                total_batch += 1
            scheduler.step()
        writer.close()
        end_time = time.time()
        print("Train Time : {:.3f} min".format(((float)((end_time - start_time)) / 60)))


    def test(self):

        self.model.load_state_dict(torch.load(self.model_saved))
        self.model.to(self.device)
        self.model.eval()
        def SquaredErrorMedian(result, predict):
            lossList = [float((result[i] - predict[i])**2) for i in range(len(result))]
            if len(lossList)%2 == 1:
                return lossList[int(len(lossList)/2)]
            else:
                return (lossList[int(len(lossList)/2)-1] + lossList[int(len(lossList)/2)])/2
        def AbsoluteErrorMedian(result, predict):
            lossList = [abs(float(result[i] - predict[i])) for i in range(len(result))]
            if len(lossList)%2 == 1:
                return lossList[int(len(lossList)/2)]
            else:
                return (lossList[int(len(lossList)/2)-1] + lossList[int(len(lossList)/2)])/2

        maeDf = np.array([], dtype=float)
        mseDf = np.array([], dtype=float)
        AEMedianDf = np.array([], dtype=float)
        SEMedianDf = np.array([], dtype=float)
        r2scoreDf = np.array([], dtype=float)
        dateDf = np.array([], dtype=str)
        print("TEST...")
        with torch.no_grad():
            for trainsData in tqdm(self.DataLoaders[1]):
                trains = trainsData['train'].to(self.device)
                results = trainsData['result'].to(self.device)
                predicts = self.model((trains, results))
                results = results.cpu().detach().numpy()
                predicts = predicts.cpu().detach().numpy()
                for index in range(results.shape[0]):
                    dates = [date[index] for date in trainsData['date']]
                    date = dates[0]
                    dates = [datetime.datetime.strptime(date, "%d.%m.%Y %H:%M:%S") for date in dates]
                    result = results[index]
                    predict = predicts[index]
                    mseDf = np.append(mseDf, mean_squared_error(result, predict))
                    maeDf = np.append(maeDf, mean_absolute_error(result, predict))
                    AEMedianDf = np.append(AEMedianDf, AbsoluteErrorMedian(result, predict))
                    SEMedianDf = np.append(SEMedianDf, SquaredErrorMedian(result, predict))
                    r2scoreDf = np.append(r2scoreDf, r2_score(result, predict))
                    dateDf = np.append(dateDf, date)
                    plt.style.use('ggplot')
                    # plt.figure(figsize=(20, 5))
                    plt.xlabel("Date")
                    plt.ylabel("Temperature")
                    plt.plot_date(dates, result, '-', label='ground truth')
                    plt.plot_date(dates, predict, '-', label='predict')
                    plt.xticks(rotation=45)
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(self.images_saved + str(date) +'.png')
                    plt.show()
        resultDf = np.column_stack((dateDf, maeDf, AEMedianDf, mseDf, SEMedianDf, r2scoreDf))
        plt.style.use('ggplot')
        plt.xlabel("Sample")
        plt.ylabel("Number")
        sampleList = [i for i in range(dateDf.shape[0])]
        plt.plot(sampleList, maeDf, '-', label='Mean Absolute Error')
        plt.plot(sampleList, AEMedianDf, '-', label='Median Absolute Error')
        plt.plot(sampleList, mseDf, '-', label='Mean Squared Error')
        plt.plot(sampleList, SEMedianDf, '-', label='Median Squared Error')
        plt.plot(sampleList, r2scoreDf, '-', label='R2Score')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.images_saved + 'error.png')
        plt.show()
        df = pd.DataFrame(resultDf, columns=['Date', 'Mean Absolute Error', 'Median Absolute Error', 'Mean Squared Error', 'Median Squared Error', 'R2Score'])
        df.to_csv(self.output_result, index=False)

