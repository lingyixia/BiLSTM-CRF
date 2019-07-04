import tensorflow as tf
import numpy as np
from Model import *
import os, functools, argparse, json
from pathlib import Path
from dataHelper import *

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='BiLSTM-CRF超参数设置')
parser.add_argument('--maxLength', type=int, default=115, help='序列最大长度')
parser.add_argument("--embeddingSize", type=int, default=64, help='字向量维度')
parser.add_argument("--hiddenSize", type=int, default=128, help='LSTM隐藏层维度')
parser.add_argument('--learnRate', type=float, default=0.005, help='学习率设置')
parser.add_argument("--dropout", type=float, default=0.5, help='dropout keep prob')
parser.add_argument("--dataDir", type=str, default='data', help='数据路径')
parser.add_argument("--batchSize", type=int, default=32, help='batchSize')
parser.add_argument("--epochNum", type=int, default=100, help='epoch数量')
parser.add_argument("--crf", type=bool, default=True, help='是否使用crf')
parser.add_argument('--l2_reg_lambda', type=float, default=0.1, help='l2正则项系数')
parser.add_argument('--modelPath', type=str, default='model', help='模型存放位置')


def model_fn(features, labels, mode, params):
    inputX, sentenceLengths = features
    model = BiLSTMCrf(inputX, labels, sentenceLengths, params['numClasses'], params['vocabSize'],
                      FLAGS.embeddingSize, FLAGS.hiddenSize, FLAGS.learnRate, FLAGS.maxLength, FLAGS.l2_reg_lambda,
                      FLAGS.dropout if mode == tf.estimator.ModeKeys.TRAIN else 1.0, FLAGS.crf)
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss, train_op, logits, sequence = model.getResult(mode)
        train_logging_hook = tf.train.LoggingTensorHook({"logits": logits, 'sequence': sequence}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        loss, metrics = model.getResult(mode)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)
    else:
        sequence = model.getResult(mode)
        predictions = {'sentence': inputX, 'sentenceLength': sentenceLengths, 'tag': sequence}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    np.set_printoptions(threshold=np.inf)
    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120, keep_checkpoint_max=5)
    dataHelper = DataHelper('./data/train.txt')
    if not Path(FLAGS.modelPath).exists():
        Path(FLAGS.modelPath).mkdir()
    with Path(FLAGS.modelPath).joinpath('params').open('w') as writer:
        json.dump(vars(FLAGS), writer)
    params = {'numClasses': len(dataHelper.index2tag), 'vocabSize': len(dataHelper.index2vocab)}
    model = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=FLAGS.modelPath, config=cfg)
    train_inputFun = functools.partial(dataHelper.input_fn, os.path.join(FLAGS.dataDir, 'train.txt'),
                                       batch_size=FLAGS.batchSize, epoch_num=FLAGS.epochNum)
    eval_inputFun = functools.partial(dataHelper.input_fn, os.path.join(FLAGS.dataDir, 'dev.txt'),
                                      batch_size=FLAGS.batchSize, is_shuffle_and_repeat=False)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inputFun)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inputFun, throttle_secs=120)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    test_inputFun = functools.partial(dataHelper.input_fn, os.path.join(FLAGS.dataDir, 'test.txt'),
                                      batch_size=FLAGS.batchSize, is_shuffle_and_repeat=False)
    predictions = model.predict(test_inputFun)
    for result in predictions:
        result = dataHelper.indexToText(result['sentence'], result['sentenceLength'], result['tag'])
        print(result)
