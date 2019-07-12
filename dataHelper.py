# coding:utf-8

# -------------------------------------------------------------------------------
# @Author        chenfeiyu01
# @Name:         dataHelper.py
# @Project       BiLSTM-CRF
# @Product       PyCharm
# @DateTime:     2019-07-04 10:05
# @Contact       chenfeiyu01@baidu.com
# @Version       1.0
# @Description:  
# -------------------------------------------------------------------------------
import functools, pickle
import tensorflow as tf
from pathlib import Path

tf.enable_eager_execution()


class DataHelper(object):
    def __init__(self, train_file):
        self.__makeVocabulary(train_file=train_file, savePath='dataProcess')

    def __makeVocabulary(self, train_file, savePath):
        if Path(savePath).joinpath('vocab.pkl').exists() and Path(savePath).joinpath('tag.pkl').exists():
            with Path(savePath).joinpath('vocab.pkl').open('rb') as reader:
                vocabDic = pickle.load(reader)
                self.vocab2index = vocabDic['vocab2index']
                self.index2vocab = vocabDic['index2vocab']
                # self.index2vocab = dict(zip(map(int, index2vocab.keys()), index2vocab.values()))
            with Path(savePath).joinpath('tag.pkl').open('rb') as reader:
                tagedDic = pickle.load(reader)
                self.tag2index = tagedDic['tag2index']
                self.index2tag = tagedDic['index2tag']
                # self.index2tag = dict(zip(map(int, index2tag.keys()), index2tag.values()))
                return
        vocabSet = set()
        tagedSet = set()
        vocabSet.add('<UNK>')
        vocabSet.add('<PAD>')
        tagedSet.add('<PAD>')
        for line in Path(train_file).open():
            line = line.strip('/n').split()
            for item in line:
                item = item.split('/')
                vocabSet.add(item[0])
                tagedSet.add(item[1])
        tagedList = list(tagedSet)
        self.vocab2index = dict(zip(vocabSet, range(len(vocabSet))))
        self.index2vocab = dict(zip(range(len(vocabSet)), vocabSet))
        self.tag2index = dict(zip(tagedList, range(len(tagedList))))
        self.index2tag = dict(zip(range(len(tagedList)), tagedList))
        vocabDic = {'vocab2index': self.vocab2index, 'index2vocab': self.index2vocab}
        tagedDic = {'tag2index': self.tag2index, 'index2tag': self.index2tag}
        with open(Path(savePath).joinpath('vocab.pkl'), mode='wb') as writer:
            pickle.dump(vocabDic, writer)
        with open(Path(savePath).joinpath('tag.pkl'), mode='wb') as writer:
            pickle.dump(tagedDic, writer)

    def __parse_fn(self, line):
        items = line.strip('/n').split()
        sentence = list()
        tag = list()
        for item in items:
            item = item.split('/')
            sentence.append(self.vocab2index.get(item[0], self.vocab2index['<UNK>']))
            tag.append(self.tag2index[item[1]])
        return (sentence, len(sentence)), tag

    def __generator_fn(self, train_file):
        with Path(train_file).open('r') as reader:
            for line in reader:
                yield self.__parse_fn(line)

    def input_fn(self, train_file, epoch_num=1, batch_size=32, is_shuffle_and_repeat=True):
        shapes = (([None], ()), [None])
        types = ((tf.int32, tf.int32), tf.int32)
        defaults = ((self.vocab2index['<PAD>'], 0), self.tag2index['<PAD>'])
        dataset = tf.data.Dataset.from_generator(
            functools.partial(self.__generator_fn, train_file),
            output_shapes=shapes,
            output_types=types)
        if is_shuffle_and_repeat:
            dataset = dataset.shuffle(1000).repeat(epoch_num)
        dataset = (dataset.padded_batch(batch_size, shapes, defaults).prefetch(1))
        return dataset

    def indexToText(self, sentence, sentenceLength, tag):
        sentence = list(map(lambda x: self.index2vocab[x], sentence[:sentenceLength]))
        print(''.join(sentence))
        tag = list(map(lambda x: self.index2tag[x], tag[:sentenceLength]))
        per, loc, org = '', '', ''


        for s, t in zip(sentence, tag):
            if t in ('B-PER', 'I-PER','E-PER'):
                per += ' ' + s if (t == 'B-PER') else s
            if t in ('B-ORG', 'I-ORG','E-ORG'):
                org += ' ' + s if (t == 'B-ORG') else s
            if t in ('B-LOC', 'I-LOC','E-LOC'):
                loc += ' ' + s if (t == 'B-LOC') else s
        return ['Person:' + per, 'Location:' + loc, 'Organzation:' + org]


# if __name__ == '__main__':
#     helper = DataHelper('data/train.txt')
#     dataset = helper.input_fn('data/train.txt')
#     for d in dataset:
#         pass
