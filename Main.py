import tensorflow as tf
import numpy as np
from dataGenerator import *
from dataProcess import *
from Model import *
import os, functools, argparse, json
from pathlib import Path

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='BiLSTM-CRF超参数设置')
parser.add_argument('--maxLength', type=int, default=115, help='序列最大长度')
parser.add_argument("--embeddingSize", type=int, default=64, help='字向量维度')
parser.add_argument("--hiddenSize", type=int, default=128, help='LSTM隐藏层维度')
parser.add_argument('--learnRate', type=float, default=0.1, help='学习率设置')
parser.add_argument("--dropout", type=float, default=0.5, help='dropout keep prob')
parser.add_argument("--dataDir", type=str, default='data', help='数据路径')
parser.add_argument("--batchSize", type=int, default=32, help='batchSize')
parser.add_argument("--epochNum", type=int, default=100, help='batchSize')
parser.add_argument("--crf", type=bool, default=True, help='是否使用crf')
parser.add_argument('--l2_reg_lambda', type=float, default=0.1, help='l2正则项系数')
parser.add_argument('--modelPath', type=str, default='model', help='模型存放位置')


def model_fn(features, labels, mode, params):
    inputX = features['sentences']
    sentenceLengths = features['sentenceLengths']
    model = BiLSTMCrf(inputX, labels, sentenceLengths, params['numClasses'], params['vocabSize'],
                      FLAGS.embeddingSize, FLAGS.hiddenSize, FLAGS.learnRate, FLAGS.maxLength, FLAGS.l2_reg_lambda,
                      FLAGS.dropout if mode == tf.estimator.ModeKeys.TRAIN else 1.0, FLAGS.crf)
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss, train_op = model.getResult(mode)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        loss, metrics = model.getResult(mode)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)
    else:
        sequence = model.getResult(mode)
        predictions = {'sentence': inputX, 'tags': sequence}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    vocab2index, index2vocab, taged2index, index2taged = makeVocabulary()
    params = {'numClasses': len(index2taged), 'vocabSize': len(index2vocab)}
    dataGenerator = DataGenerator(vocab2index, index2vocab, taged2index, index2taged, FLAGS.maxLength)
    if not Path(FLAGS.modelPath).exists():
        Path(FLAGS.modelPath).mkdir()
    with Path(FLAGS.modelPath).joinpath('params').open('w') as writer:
        json.dump(vars(FLAGS), writer)
    model = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=FLAGS.modelPath)
    train_inputFun = functools.partial(dataGenerator.input_fn, os.path.join(FLAGS.dataDir, 'train.txt'),
                                       batchSize=FLAGS.batchSize, epochNum=FLAGS.epochNum)
    model.train(train_inputFun)
    eval_inputFun = functools.partial(dataGenerator.input_fn, os.path.join(FLAGS.dataDir, 'dev.txt'),
                                      batchSize=FLAGS.batchSize, ifShuffleAndRepeat=False)
    model.evaluate(eval_inputFun)
    test_inputFun = functools.partial(dataGenerator.input_fn, os.path.join(FLAGS.dataDir, 'test.txt'),
                                      batchSize=FLAGS.batchSize,
                                      ifShuffleAndRepeat=False)
    predictions = model.predict(test_inputFun)
    for result in predictions:
        sentence, tags = dataGenerator.indexToText(result['sentence'], result['tags'])
        print(dict(zip(sentence, tags)))
