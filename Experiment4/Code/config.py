from gensim.models import FastText
import numpy as np

class config(object):
    """
    定义相关配置
    """

    def __init__(self, args):
        super(config, self).__init__()

        # 数据相关路径
        self.data_path = '../Dataset'
        self.data_type = args['data']

        # 训练相关
        self.model = args['model']
        self.device = args['dev']
        self.batch_size = int(args['bs'])
        self.optimizer = args['opt']
        self.epoch = int(args['epoch'])
        self.learning_rate = float(args['lr'])
        self.maxiter_without_improvement = 1000
        self.log_path = '../Models/Logs/'
        self.model_saved = '../Models/Model/'

        # 模型参数
        # 梯度裁剪
        self.theta = 2
        self.num_layers = 2
        self.dropout = 0.5


        if self.data_type == 'Text':
            self.hidden_size = 128
            self.class_num = 10
            self.input_path = self.data_path + '/online_shopping_10_cats/online_shopping_10_cats.csv'
            self.pretrained = '../Models/pretrained/FastText.bin'
            self.pretrained_model = FastText.load(self.pretrained)
            self.key_to_index = self.pretrained_model.wv.key_to_index
            self.index_to_key = self.pretrained_model.wv.index_to_key
            # 词典大小
            self.vocab_size = len(self.index_to_key)
            # 每个词向量的维度
            self.vector_size = 300
            self.weight = np.zeros((self.vocab_size, self.vector_size))
            for i in range(self.vocab_size):
                self.weight[i][:] = self.pretrained_model.wv[self.index_to_key[i]]
            # 句子长度
            self.padding_size = 64
            self.images_saved = '../Models/Images/'
        else:
            self.input_size = 21
            self.hidden_size = 256
            self.output_size = 288
            self.train_path = self.data_path + '/jena_climate_2009_2016/train.csv'
            self.test_path = self.data_path + '/jena_climate_2009_2016/test.csv'
            self.images_saved = '../Models/TemperatureOutput/'
            self.output_result = '../Models/TemperatureOutput/output.csv'


