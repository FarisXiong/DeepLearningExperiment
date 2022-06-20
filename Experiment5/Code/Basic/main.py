from argparser import Parser
from config import config
from trainer import Trainer
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    parser = Parser()
    config = config(parser.args)
    Trainer = Trainer(config)
    Trainer.train()




