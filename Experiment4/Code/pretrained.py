from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
import jieba
import pandas as pd
from tqdm import tqdm

class config():
    def __init__(self):
        self.vector_size = 300
        self.windows = 5
        self.min_count = 1
        self.seed = 2
        self.epoch = 10
        self.input_path = './Experiment4/Dataset/online_shopping_10_cats/online_shopping_10_cats.csv'
        self.model_saved = './Experiment4/Models/pretrained/FastText.bin'
        

def getTextData(config):
    """
    返回训练语料库
    """
    Data = pd.read_csv(config.input_path)
    corpusSentence = list(Data['review'])
    # corpus = [[word for word in jieba.cut(str(sentence), cut_all=False)] for sentence in tqdm(corpusSentence)]
    corpus = [[word for word in jieba.cut(str(sentence), cut_all=False)] for sentence in corpusSentence]
    corpus.extend([['[UNK]'],['[PAD]']])
    return corpus


if __name__ == '__main__':
    config = config()
    corpus = getTextData(config)
    model = FastText(vector_size=config.vector_size, window=config.windows, min_count=config.min_count, seed=config.seed)
    model.build_vocab(corpus_iterable=corpus)
    model.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=config.epoch)
    model.save(config.model_saved)

