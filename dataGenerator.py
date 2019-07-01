import numpy as np
import tensorflow as tf
from tensorflow.contrib import keras


class DataGenerator:
    def __init__(self, vocab2index, index2vocab, taged2index, index2taged, maxLength):
        self.__vocab2index = vocab2index
        self.__index2vocab = index2vocab
        self.__taged2index = taged2index
        self.__index2taged = index2taged
        self.__maxLength = maxLength

    def __getData(self, inputFile):
        sentences = list()
        sentenceLengths = list()
        tags = list()
        with open(inputFile, encoding='utf-8') as reader:
            sentence = list()
            tag = list()
            for line in reader:
                line = line.replace('\n', '')
                if line == '':
                    # sentence = [self.__vocab2index['<START>']] + sentence
                    # tag = [self.__taged2index['<START>']] + tag
                    # sentence.append(self.__vocab2index['<END>'])
                    # tag.append(self.__taged2index['<END>'])
                    sentenceLengths.append(len(sentence))
                    sentences.append(sentence.copy())
                    tags.append(tag.copy())
                    sentence.clear()
                    tag.clear()
                else:
                    thisWord = line.split(' ')[0]
                    thisTag = line.split(' ')[1]
                    sentence.append(self.__vocab2index.get(thisWord, self.__vocab2index['<UNK>']))
                    tag.append(self.__taged2index[thisTag])
            sentences = keras.preprocessing.sequence.pad_sequences(sentences, self.__maxLength, padding='post',
                                                                   value=[self.__vocab2index['<PAD>']])
            tags = keras.preprocessing.sequence.pad_sequences(tags, self.__maxLength, padding='post',
                                                              value=self.__taged2index['<PAD>'])
        return sentences, tags, np.asarray(sentenceLengths)

    def input_fn(self, inputFile, batchSize, epochNum=1, ifShuffleAndRepeat=True):
        sentences, tages, sentenceLengths = self.__getData(inputFile)
        dataset = tf.data.Dataset.from_tensor_slices((sentences, sentenceLengths, tages))
        if ifShuffleAndRepeat:
            dataset = dataset.shuffle(1000)
            dataset = dataset.repeat(epochNum)
        dataset = dataset.batch(batchSize)
        iterator = dataset.make_one_shot_iterator()
        sentences, sentenceLengths, tag = iterator.get_next()
        sentences = {'sentences': sentences, 'sentenceLengths': sentenceLengths}
        return sentences, tag

    def indexToText(self, sentence, tag):
        sentence = list(map(lambda x: self.__index2vocab[x], sentence))
        tag = list(map(lambda x: self.__index2taged[x], tag))
        per, loc, org = '', '', ''

        for s, t in zip(sentence, tag):
            if t in ('B-PER', 'I-PER'):
                per += ' ' + s if (t == 'B-PER') else s
            if t in ('B-ORG', 'I-ORG'):
                org += ' ' + s if (t == 'B-ORG') else s
            if t in ('B-LOC', 'I-LOC'):
                loc += ' ' + s if (t == 'B-LOC') else s
        return ['Person:' + per, 'Location:' + loc, 'Organzation:' + org]
