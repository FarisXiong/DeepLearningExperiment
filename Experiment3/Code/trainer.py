from dataset import getDataset
from torch.utils.data import DataLoader
from models import *
import torch
from config import config
import time
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from sklearn import metrics
import numpy as np
import pandas as pd
from transformers import AutoFeatureExtractor, SwinForImageClassification
from transformers import Trainer
from transformers import TrainingArguments

class trainer():
    def __init__(self,config:config):
        self.trainer_ = basicTrainer(config)

    def train(self):
        self.trainer_.train()

    def test(self):
        self.trainer_.test()




class basicTrainer():
    def __init__(self, config:config):
        self.log_dir = config.log_dir
        self.epoch = config.epoch
        self.fold_k = config.fold_k
        self.device = config.device
        self.model_saved = config.model_saved
        self.maxiter_without_improvement = config.maxiter_without_improvement
        self.dataset, self.labelMaps = getDataset(config)
        self.output_path = config.output_path
        self.trainDataLoader = [DataLoader(dataset=dataset[0], batch_size=config.batch_size, shuffle=True) for dataset in self.dataset[0]]
        self.devDataLoader = [DataLoader(dataset=dataset[1], batch_size=config.batch_size, shuffle=True) for dataset in self.dataset[0]]
        self.testDataLoader = DataLoader(dataset=self.dataset[1], batch_size=config.batch_size, shuffle=False)
        self.model_name = config.model
        if config.model == 'ResNet':
            self.model = [ResNet18(config.class_num).to(self.device) for i in range(config.fold_k)]
        elif config.model == 'VGG':
            self.model = [VGG(config, True).to(self.device) for i in range(config.fold_k)]
        elif config.model == 'SENet':
            self.model = [SENet34(config.class_num).to(self.device) for i in range(config.fold_k)]
        else:
            self.model = [SwinTransformer(config.class_num).to(self.device) for i in range(config.fold_k)]
        self.learning_rate = config.learning_rate
        self.optimizer = config.optimizer
        if self.optimizer == 'SGD':
            self.optimizer = [torch.optim.SGD(model.parameters(), lr=self.learning_rate) for model in self.model]
        elif self.optimizer == 'Adam':
            self.optimizer = [torch.optim.Adam(model.parameters(), lr=self.learning_rate) for model in self.model]
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.scheduler = [torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5, last_epoch=-1) for optimizer in self.optimizer]


    def train(self):
        # 记录时间
        start_time = time.time()
        for model in self.model:
            model.train()
        dev_best_loss = [float('inf')]*self.fold_k       # 记录验证集上最低的loss
        dev_best_micro_f1score = [float(0)]*self.fold_k  # 记录验证集上最高的acc
        dev_best_macro_f1score = [float(0)]*self.fold_k  # 记录验证集上最高的f1score
        writer = [SummaryWriter(logdir=self.log_dir[i]) for i in range(self.fold_k)]
        for i in range(self.fold_k):
            flag = False # 是否结束训练
            total_batch = 0          # 记录总共训练的批次
            last_improve = 0  # 记录上一次dev的loss下降时的批次
            print("K Fold [{}/{}]".format(i+1, self.fold_k))
            for epoch in range(self.epoch):
                print("Epoch [{}/{}]".format(epoch + 1, self.epoch))
                for index, data in enumerate(self.trainDataLoader[i]):
                    trains = data['data'].to(self.device)
                    labels = data['label'].to(self.device)
                    outputs = self.model[i](trains)
                    self.model[i].zero_grad()
                    loss = F.cross_entropy(outputs, labels)
                    loss.backward()
                    self.optimizer[i].step()
                    # 输出当前效果
                    if total_batch % 10 == 0:
                        ground_truth = labels.data.cpu()
                        predict_labels = torch.argmax(outputs, dim=1).cpu().numpy()
                        train_acc = metrics.accuracy_score(ground_truth, predict_labels)
                        dev_micro_f1score, dev_macro_f1score, dev_loss = self.eval(i)
                        improve = ''
                        if dev_loss < dev_best_loss[i]:
                            dev_best_loss[i] = dev_loss
                        if dev_micro_f1score > dev_best_micro_f1score[i]:
                            dev_best_micro_f1score[i] = dev_micro_f1score
                            torch.save(self.model[i].state_dict(), self.model_saved[i])
                            improve = '*'
                            last_improve = total_batch
                        if dev_macro_f1score > dev_best_macro_f1score[i]:
                            dev_best_macro_f1score[i] = dev_macro_f1score

                        print(
                            "Iter:{:4d} TrainLoss:{:.12f} TrainAcc:{:.5f} DevLoss:{:.12f} DevMicroF1Score:{:.5f} DevMacroF1Score:{:.5f} Improve:{}".format(
                                total_batch, loss.item(), train_acc, dev_loss, dev_micro_f1score, dev_macro_f1score, improve))
                        writer[i].add_scalar("loss/train", loss.item(), total_batch)
                        writer[i].add_scalar("acc/train", train_acc, total_batch)
                        writer[i].add_scalar("loss/dev", dev_loss, total_batch)
                        writer[i].add_scalar("microf1score/dev", dev_micro_f1score, total_batch)
                        writer[i].add_scalar("macrof1score/dev", dev_macro_f1score, total_batch)
                        self.model[i].train()
                    total_batch += 1
                    if total_batch - last_improve > self.maxiter_without_improvement:
                        print("No optimization for a long time, auto-stopping...")
                        flag = True
                        break
                if flag:
                    break
                self.scheduler[i].step()
        for writer_ in writer:
            writer_.close()
        end_time = time.time()
        print("Train Time : {:.3f} min , The Best Micro F1 Score in Dev : {}, The Best Macro f1-score in Dev : {}".format(
            ((float)((end_time - start_time)) / 60), dev_best_micro_f1score, dev_best_macro_f1score))

    def eval(self, index):
        for model in self.model:
            model.eval()
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        Loss_sum = float(0)
        with torch.no_grad():
            for index_, data in enumerate(self.devDataLoader[index]):
                trains = data['data'].to(self.device)
                labels = data['label'].to(self.device)
                outputs = self.model[index](trains)
                Loss_sum += F.cross_entropy(outputs, labels).item()
                outputs = torch.argmax(outputs, dim=1).unsqueeze(1)
                ground_truth = labels.cpu().numpy()
                predict_labels = outputs.cpu().numpy()
                labels_all = np.append(labels_all, ground_truth)
                predict_all = np.append(predict_all, predict_labels)
        microf1score = metrics.f1_score(labels_all, predict_all, average='micro')
        macrof1score = metrics.f1_score(labels_all, predict_all, average='macro')
        return microf1score, macrof1score, Loss_sum

    def test(self):
        for i in range(self.fold_k):
            self.model[i].load_state_dict(torch.load(self.model_saved[i]))
        for model in self.model:
            model.eval()
        name_all = np.array([],dtype=str)
        predict_all = np.array([], dtype=int)
        with torch.no_grad():
            for index, data in enumerate(self.testDataLoader):
                trains = data['data'].to(self.device)
                outputs = None
                for model in self.model:
                    if outputs == None:
                        outputs = torch.argmax(model(trains).detach().cpu(), dim=1).unsqueeze(1)
                    else:
                        outputs = torch.cat((outputs, torch.argmax(model(trains).detach().cpu(), dim=1).unsqueeze(1)),dim=1)
                outputs = torch.mode(outputs, dim=1)[0]
                predict_all = np.append(predict_all, outputs.numpy())
                names = data['name']
                name_all = np.append(name_all, names)
        predict_result = np.column_stack((name_all, predict_all))
        df = pd.DataFrame(predict_result, columns=['file', 'species'])
        def int2map(row):
            return self.labelMaps[1][int(row['species'])]
        df['species'] = df.apply(int2map,axis=1)
        df.to_csv(self.output_path, index=False)


