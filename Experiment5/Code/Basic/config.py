import torch

class config():
    def __init__(self, args):
        # 数据路径
        self.data_path = '../../Dataset/'
        self.points_mat = self.data_path + 'points.mat'

        # 参数设置
        self.device = args['device']
        if self.device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.model = args['model']
        self.mode = args['mode']
        self.batch_size = args['bs']
        self.epoch = args['epoch']
        self.Glr = args['Glr']
        self.Dlr = args['Dlr']

        # 训练设置
        # D 训练次数
        if self.model == 'GAN':
            self.dk = 5
            self.gk = 5
        elif self.model == 'WGAN':
            self.dk = 5
            self.gk = 1
            self.clip_value = 0.01
        else:
            self.dk = 5
            self.gk = 2
            self.lambda_value = 0.1

        self.model_saved_G = '../../Trains/Models/' + self.model + '_G.pkl'
        self.model_saved_D = '../../Trains/Models/' + self.model + '_D.pkl'
        self.log_path = '../../Trains/Logs/' + self.model
        self.images_saved = '../../Trains/Images/' + self.model + '/'






