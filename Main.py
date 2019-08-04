import tensorflow as tf
import numpy as np
from dataHelper import *
from Model import *
import os, functools, argparse, json
from pathlib import Path

tf.logging.set_verbosity(tf.logging.INFO)

parser = argparse.ArgumentParser(description='BiLSTM-CRF超参数设置')
parser.add_argument('--maxLength', type=int, default=150, help='序列最大长度')
parser.add_argument("--embeddingSize", type=int, default=100, help='字向量维度')
parser.add_argument("--hiddenSize", type=int, default=128, help='LSTM隐藏层维度')
parser.add_argument('--base_learn_rate', type=float, default=0.0015, help='学习率设置')
parser.add_argument("--dropout", type=float, default=0.5, help='dropout5keep prob')
parser.add_argument("--data_path", type=str, default='data', help='数据路径')
parser.add_argument("--batchSize", type=int, default=64, help='batchSize')
parser.add_argument("--epochNum", type=int, default=100, help='batch数量')
parser.add_argument("--crf", type=bool, default=True, help='是否使用crf')
parser.add_argument('--l2_reg_lambda', type=float, default=0.001, help='l2正则项系数')
parser.add_argument('--modelPath', type=str, default='model', help='模型存放位置')
parser.add_argument('--random_seed', type=int, default=1234, help='初始化种子')
parser.add_argument('--encode_layer_size', type=bool, default=2, help='编码层bisltm层数')


def model_fn(features, labels, mode, params):
    inputX, sentenceLengths, labels = features
    model = BiLSTMCrf(inputX, labels, sentenceLengths, params['numClasses'], params['vocabSize'],
                      FLAGS.embeddingSize, FLAGS.hiddenSize, FLAGS.base_learn_rate, FLAGS.maxLength,
                      FLAGS.l2_reg_lambda,
                      FLAGS.dropout if mode == tf.estimator.ModeKeys.TRAIN else 1.0, FLAGS.crf,
                      layer_size=params['encodeLayerSize'], eval_tags=params['eval_tags'],
                      random_seed=params['random_seed'])
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss, train_op, learn_rate = model.getResult(mode)
        train_logging_hook = tf.train.LoggingTensorHook({"learn_rate": learn_rate}, every_n_iter=100)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=[train_logging_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
        loss, metrics, predictions = model.getResult(mode)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)
    else:
        sequence = model.getResult(mode)
        predictions = {'sentence': inputX, 'sentenceLength': sentenceLengths, 'tag': sequence, 'labels': labels}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120, keep_checkpoint_max=5)
    # os.chdir("/content/drive/ColaboratoryLab/NER/达观杯")
    dataHelper = DataHelper(os.path.join(FLAGS.data_path, 'train.txt'))
    eval_tags = dataHelper.tag2index.copy()
    print(eval_tags)
    eval_tags.pop('<PAD>')
    eval_tags.pop('O')
    eval_tags = list(eval_tags.values())
    if not Path(FLAGS.modelPath).exists():
        Path(FLAGS.modelPath).mkdir()
    with Path(FLAGS.modelPath).joinpath('params').open('w') as writer:
        json.dump(vars(FLAGS), writer)
    params = {'numClasses': len(dataHelper.index2tag),
              'vocabSize': len(dataHelper.index2vocab),
              'encodeLayerSize': FLAGS.encode_layer_size,
              'eval_tags': eval_tags,
              'random_seed': FLAGS.random_seed}
    model = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=FLAGS.modelPath, config=cfg)
    train_inputFun = functools.partial(dataHelper.input_fn, os.path.join(FLAGS.data_path, 'train.txt'),
                                       batch_size=FLAGS.batchSize, epoch_num=FLAGS.epochNum)
    eval_inputFun = functools.partial(dataHelper.input_fn, os.path.join(FLAGS.data_path, 'dev.txt'),
                                      batch_size=FLAGS.batchSize, is_shuffle_and_repeat=False)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inputFun)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inputFun, steps=None)
    results = tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    test_inputFun = functools.partial(dataHelper.input_fn, os.path.join(FLAGS.data_path, 'realtest.txt'),
                                      batch_size=FLAGS.batchSize, is_shuffle_and_repeat=False)
    predictions = model.predict(test_inputFun)
    for result in predictions:
        result = dataHelper.indexToText(result['sentence'], result['sentenceLength'], result['tag'], result['labels'])
        with open('resulttags.txt', mode='a') as writer:
            json.dump(result, writer)
            writer.write('\n')
        # print(result)
        # print()
