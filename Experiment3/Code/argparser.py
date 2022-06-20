import argparse


class Parser:
    """命令行参数解析器"""

    def __init__(self):
        # 命令行参数 key 对应所有的合法值 value
        # 合法值 value 的第一个值 value[0] 是参数的默认值
        self.arg_available_values = {
            'aug': ('none', 'all'),
            'dev': ('cuda', 'cpu'),
            'model': ('SENet', 'VGG', 'ResNet', 'SwinTransformer'),
            'mode': ('train', 'test', 'both'),
            'opt': ('Adam', 'SGD'),
            'lr': (1e-4,),
            'bs': (128,),
            'epoch': (1000,),
            'kf': (1,)
        }

        # 命令行参数提示信息
        self.arg_help = {
            'aug': 'data augmentation',
            'dev': 'cuda/cpu',
            'model': 'VGG/ResNet/SENet/SwinTransformer',
            'mode': 'train/test/both',
            'opt': 'SGD/Adam',
            'lr': 'learning rate',
            'bs': 'batch size',
            'epoch': 'epoch',
            'kf': 'k fold'
        }

        # 命令行参数类型
        self.arg_type = {
            'aug': str,
            'dev': str,
            'model': str,
            'mode': str,
            'opt': str,
            'lr': float,
            'bs': int,
            'epoch': int,
            'kf': int
        }

        # 接收到的命令行参数，初始值是默认值
        self.args = {}
        for k, v in self.arg_available_values.items():
            self.args[k] = v[0]

        self.parse()

    def parse(self):
        """解析命令行参数，将参数合法化，并存放到 self.args 中"""

        p = argparse.ArgumentParser(description='基于 PyTorch 实现 VGG/ResNet/SENet 等结构')

        for arg, arg_values in self.arg_available_values.items():
            help_msg = self.arg_help[arg] + f' (default: {str(self.args[arg])})'
            p.add_argument(f'--{arg}', type=self.arg_type[arg], default=self.args[arg], help=help_msg)

        args = p.parse_args()

        self.legalise(args)

    def legalise(self, args):
        """命令行参数合法化，对不合法的值使用默认值进行覆盖"""

        for arg, arg_type in self.arg_type.items():
            if arg_type == str:
                if vars(args)[arg] in self.arg_available_values[arg]:
                    # 解析出的参数的值在合法值集合中
                    # 添加到结果
                    self.args[arg] = vars(args)[arg]
                else:
                    # 解析出的参数的值不在合法值集合中
                    # 输出错误信息，使用参数默认值
                    print(f'cannot recognize {arg} = `{vars(args)[arg]}`, using default `{self.args[arg]}`')
            elif arg_type == float or arg_type == int:
                if vars(args)[arg] > 0:
                    # 解析出的参数的值合法
                    # 添加到结果
                    self.args[arg] = vars(args)[arg]
                else:
                    # 解析出的参数的值不合法
                    # 输出错误信息，使用参数默认值
                    print(f'cannot use {arg} = `{vars(args)[arg]}`, using default `{self.args[arg]}`')
            else:
                print("===== ARGUMENT ERROR =====")