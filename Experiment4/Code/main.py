from argparser import Parser
from config import config
from trainer import trainer
import os

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = Parser()
    config = config(parser.args)
    trainer = trainer(config)
    # trainer.train()
    trainer.test()






