import os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from model import config
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class dataList():
    def __init__(self, config):
        self.data_path = config.data_path
        (self.train,self.dev,self.test),self.labelMap = self.getdata()

    def getdata(self):
        trainList = []
        devList = []
        testList = []
        filesDir =[filedir for filedir in os.listdir(self.data_path)]
        assert len(filesDir) == 101
        labelMapL2N = {}
        labelMapN2L = {}
        for fileDir in filesDir:
            labelMapN2L[len(labelMapL2N)] = fileDir
            labelMapL2N[fileDir] = len(labelMapL2N)
            path = os.path.join(self.data_path,fileDir)
            files = [os.path.join(path,file) for file in os.listdir(path)]
            train,test = train_test_split(files,train_size=0.8, random_state=2)
            dev,test = train_test_split(test,test_size=0.5, random_state=2)
            # 分割训练集、验证集、测试集
            trainList.extend([(trainItem,labelMapL2N[fileDir]) for trainItem in train])
            devList.extend([(devItem,labelMapL2N[fileDir]) for devItem in dev])
            testList.extend([(testItem,labelMapL2N[fileDir]) for testItem in test])
        return (trainList,devList,testList),labelMapN2L

class myDataset(Dataset):
    def __init__(self,datasetList,config):
        self.transforms =transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),     # 缩放
            transforms.ToTensor(),             # 图片转张量
            transforms.Normalize((config.mean1,config.mean2,config.mean3),(config.std1,config.std2,config.std3))
        ])
        # self.dataset = [(self.transforms(Image.open(data[0]).convert('RGB')),data[1]) for data in datasetList]
        self.dataset = [(self.transforms(Image.open(data[0]).convert('RGB')),data[1]) for data in tqdm(datasetList)]

    def __getitem__(self, item):
        dataDict = {
            'data':self.dataset[item][0],
            'label':self.dataset[item][1]
        }
        return dataDict

    def __len__(self):
        return len(self.dataset)







