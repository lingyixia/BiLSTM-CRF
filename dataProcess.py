import json, os, pickle


def makeVocabulary(originPath='data/train.txt', vocabPath='dataProcess'):
    if os.path.exists(os.path.join(vocabPath, 'vocab.pkl')) and os.path.exists(os.path.join(vocabPath, 'taged.pkl')):
        return getVocabAndTags(vocabPath)
    vocabSet = set()
    tagedSet = set()
    vocabSet.add('<UNK>')
    vocabSet.add('<PAD>')
    vocabSet.add('<START>')
    vocabSet.add('<END>')
    tagedSet.add('<START>')
    tagedSet.add('<END>')
    trainFile = open(originPath, encoding='utf-8')
    for line in trainFile:
        line = line.replace('\n', '')
        if line != '':
            line = line.split(' ')
            vocabSet.add(line[0])
            tagedSet.add(line[1])
    tagedList = list(tagedSet)
    vocab2index = dict(zip(vocabSet, range(len(vocabSet))))
    index2vocab = dict(zip(range(len(vocabSet)), vocabSet))
    taged2index = dict(zip(tagedList, range(len(tagedList))))
    index2taged = dict(zip(range(len(tagedList)), tagedList))
    vocabDic = {'vocab2index': vocab2index, 'index2vocab': index2vocab}
    tagedDic = {'taged2index': taged2index, 'index2taged': index2taged}
    with open(os.path.join(vocabPath, 'vocab.pkl'), mode='wb') as writer:
        pickle.dump(vocabDic, writer)
    with open(os.path.join(vocabPath, 'taged.pkl'), mode='wb') as writer:
        pickle.dump(tagedDic, writer)
    return vocab2index, index2vocab, taged2index, index2taged


def getVocabAndTags(path='dataProcess'):
    with open(os.path.join(path, 'vocab.pkl'), mode='rb') as reader:
        vocabDic = pickle.load(reader)
        vocab2index = vocabDic['vocab2index']
        index2vocab = vocabDic['index2vocab']
        index2vocab = dict(zip(map(int, index2vocab.keys()), index2vocab.values()))
    with open(os.path.join(path, 'taged.pkl'), mode='rb') as reader:
        tagedDic = pickle.load(reader)
        taged2index = tagedDic['taged2index']
        index2taged = tagedDic['index2taged']
        index2taged = dict(zip(map(int, index2taged.keys()), index2taged.values()))
    return vocab2index, index2vocab, taged2index, index2taged
