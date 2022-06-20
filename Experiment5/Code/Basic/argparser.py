import argparse


class Parser:
    """命令行参数解析器"""

    def __init__(self):
        # 命令行参数 key 对应所有的合法值 value
        # 合法值 value 的第一个值 value[0] 是参数的默认值
        self.arg_available_values = {
            'device': ['cuda', 'cpu'],
            'model': ['GAN', 'WGAN', 'WGAN-GP'],
            'mode': ['both', 'train'],
            'Glr': [1e-4, ],
            'Dlr': [1e-4, ],
            'bs': [128, ],
            'epoch': [1000, ],
        }

        # 命令行参数提示信息
        self.arg_help = {
            'device': 'cuda/cpu',
            'model': 'GAN/WGAN/WGAN-GP',
            'mode': 'train/test/both',
            'Glr': 'generate learning rate',
            'Dlr': 'discriminator learning rate',
            'bs': 'batch size',
            'epoch': 'epoch',
        }

        # 命令行参数类型
        self.arg_type = {
            'device': str,
            'model': str,
            'mode': str,
            'Glr': float,
            'Dlr': float,
            'bs': int,
            'epoch': int,
        }

        # 接收到的命令行参数，初始值是默认值
        self.args = {}
        for k, v in self.arg_available_values.items():
            self.args[k] = v[0]

        self.parse()

    def parse(self):
        """解析命令行参数，将参数合法化，并存放到 self.args 中"""

        p = argparse.ArgumentParser(description='基于PyTorch实现GAN、WGAN、WGAN-GP')

        for arg, arg_values in self.arg_available_values.items():
            help_msg = self.arg_help[arg] + f' (default: {str(self.args[arg])})'
            p.add_argument(f'--{arg}', type=self.arg_type[arg], default=self.args[arg], help=help_msg)
        args = p.parse_args()
        self.legalise(args)

    def legalise(self, args):
        """命令行参数合法化，对不合法的值使用默认值进行覆盖"""

        for arg, arg_type in self.arg_type.items():
            if vars(args)[arg] == 'None':
                self.args[arg] = vars(args)[arg]  # 'None'
            else:
                if self.arg_type[arg] == str and vars(args)[arg] in self.arg_available_values[arg]:
                    self.args[arg] = vars(args)[arg]
                elif self.arg_type[arg] == int or self.arg_type[arg] == float:
                    self.args[arg] = vars(args)[arg]
                else:
                    # 解析出的参数的值不在合法值集合中
                    # 输出错误信息，使用参数默认值
                    print(f'cannot recognize {arg} = `{vars(args)[arg]}`, using default `{self.args[arg]}`')
