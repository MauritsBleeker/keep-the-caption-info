"""
Create a vocabulary wrapper.
Original code:
https://github.com/yalesong/pvse/blob/master/vocab.py
https://github.com/naver-ai/pcme/blob/main/datasets/vocab.py
"""

import pickle
from nltk.tokenize import word_tokenize
from collections import Counter
import logging


class Vocabulary(object):
    """
    Simple vocabulary wrapper.
    """

    def __init__(self):
        self.idx = 0
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def load_from_pickle(self, data_path):
        with open(data_path, 'rb') as fin:
            data = pickle.load(fin)
        self.idx = data['idx']
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(captions, threshold=0):
    """
    Build a simple vocabulary wrapper.
    :param captions:
    :param threshold:
    :return:
    """

    counter = Counter()

    for caption in captions:
        tokens = word_tokenize(caption.lower())
        counter.update(tokens)

    # Discard if the occurrence of the word is less than min_word_cnt.
    words = [word for word, cnt in counter.items() if cnt >= threshold]
    logging.info('Vocabulary size: {}'.format(len(words)))

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add words to the vocabulary.
    for word in words:
        vocab.add_word(word)
    return vocab
