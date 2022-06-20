from config import config
from torch.utils.data import Dataset
from sklearn import preprocessing
import scipy.io
import numpy as np
import torch


def getDataset(config):
    """
    Args:
        config: 配置

    Return:
        训练集Dataset
        训练集数据
    """
    data = scipy.io.loadmat(config.points_mat)
    # ground_truth = np.row_stack((data['a'], data['b'], data['b'], data['c']))
    ground_truth = data['xx']

    # scaler = preprocessing.MaxAbsScaler()
    # ground_truth = scaler.fit_transform(ground_truth)

    trainDataset = myDataset(ground_truth)
    return trainDataset, np.transpose(ground_truth)


class myDataset(Dataset):
    def __init__(self, samples):
        super(myDataset, self).__init__()
        self.samples = samples

    def __getitem__(self, item):
        sample = self.samples[item]
        return {
            'data': torch.Tensor(sample)
        }


    def __len__(self):
        return self.samples.shape[0]




