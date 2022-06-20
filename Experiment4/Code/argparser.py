import argparse
import torch


class Parser:
    """命令行参数解析器"""

    def __init__(self):
        # 接收到的命令行参数，初始值是默认值
        self.args = {
            'data': 'Text',
            'dev': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'model': 'LSTM',
            'opt': 'Adam',
            'lr': 1e-5,
            'bs': 128,
            'epoch': 100
        }

        self.parse()

    def parse(self):
        """解析命令行参数，将参数合法化，并存放到 self.args 中"""

        p = argparse.ArgumentParser(description='基于RNN、GRU、LSTM实现的文本分类、天气预测')

        p.add_argument('--data', type=str, default='Text', help='data augmentation (default: True)')
        p.add_argument('--dev', type=str, default='cuda', help='cuda or cpu (default: cuda)')
        p.add_argument('--model', type=str, default='BiLSTM', help='VGG/ResNet/SENet (default: SENet)')
        p.add_argument('--opt', type=str, default='Adam', help='SGD/Adam (default: Adam)')
        p.add_argument('--lr', type=float, default=1e-5, help='learning rate (default: 1e-5)')
        p.add_argument('--bs', type=int, default=128, help='batch size (default: 128)')
        p.add_argument('--epoch', type=int, default=1000, help='max epoch (default: 10000)')

        args = p.parse_args()

        self.legalise(args)

    def legalise(self, args):
        """命令行参数合法化，对不合法的值使用默认值进行覆盖"""

        if args.data in ('Text', 'Weather'):
            self.args['data'] = str(args.data)
        else:
            print(f'cannot recognize aug = `{args.data}`, using default `{str(self.args["data"])}`')

        if args.dev == 'cuda':
            self.args['dev'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif args.dev == 'cpu':
            self.args['dev'] = torch.device(args.dev)
        else:
            print(f'cannot recognize dev = `{args.dev}`, using default `{str(self.args["dev"])}`')

        if args.model in ('RNN', 'GRU', 'LSTM', 'BiLSTM'):
            self.args['model'] = args.model
        else:
            print(f'cannot recognize model = `{args.model}`, using default `{str(self.args["model"])}`')

        if args.opt in ('Adam', 'SGD'):
            self.args['opt'] = args.opt
        else:
            print(f'cannot recognize opt = `{args.opt}`, using default `{str(self.args["opt"])}`')

        if args.lr > 0:
            self.args['lr'] = args.lr
        else:
            print(f'cannot use lr = `{args.lr}`, using default `{str(self.args["lr"])}`')

        if args.bs > 0:
            self.args['bs'] = args.bs
        else:
            print(f'cannot use bs = `{args.bs}`, using default `{str(self.args["bs"])}`')

        if args.epoch > 0:
            self.args['epoch'] = args.epoch
        else:
            print(f'cannot use epoch = `{args.epoch}`, using default `{str(self.args["epoch"])}`')

        if args.data == 'Weather' and args.model != 'LSTM':
            print(f'cannot use model = `{args.model}`, using default `{str(self.args["model"])}`')
            self.args['model'] = 'LSTM'
        if args.data == 'Weather' and args.opt != 'Adam':
            print(f'cannot use optimizer = `{args.opt}`, using default `{str(self.args["opt"])}`')
            self.args['opt'] = 'Adam'