import tensorflow as tf
import tensorflow.contrib.crf as crf
from tf_metrics import precision, recall, f1
from tensorflow.contrib import keras


class BiLSTMCrf(object):
    def __init__(self, inputX, inputY, sentenceLengths, numClasses, vocabSize, embeddingSize,
                 hiddenSize, learnRate, maxLength, l2_reg_lambda, dropout_keep_prob, crf):
        self.numClasses = numClasses
        self.vocabSize = vocabSize
        self.embeddingSize = embeddingSize
        self.hiddenSize = hiddenSize
        self.learnRate = learnRate
        self.maxLength = maxLength
        self.l2_reg_lambda = l2_reg_lambda
        self.inputX = inputX
        self.inputY = inputY
        self.sentenceLengths = sentenceLengths
        self.dropout_keep_prob = dropout_keep_prob
        self.crf = crf
        self.__addEmbeddingLayer()
        self.__addBiLSTMLayer()

    def __addEmbeddingLayer(self):
        with tf.name_scope('embeddingLayer'):
            embedding = tf.get_variable(name='embedding', shape=[self.vocabSize, self.embeddingSize],
                                        initializer=tf.contrib.layers.xavier_initializer())
            embeddingInput = tf.nn.embedding_lookup(embedding, self.inputX)
            self.embeddingInput = tf.nn.dropout(embeddingInput, rate=1 - self.dropout_keep_prob)

    def __addBiLSTMLayer(self):
        with tf.name_scope('BiLSTMLayer'):
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenSize),
                output_keep_prob=self.dropout_keep_prob)
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hiddenSize),
                output_keep_prob=self.dropout_keep_prob)
            # keras.layers.Bidirectional(
            #     keras.layers.LSTM(units=self.hiddenSize, dropout=1 - self.dropout_keep_prob)).apply(
            #     inputs=self.embeddingInput, mask=)
            bidOutput, bidCurrent_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                          cell_bw=lstm_bw_cell,
                                                                          sequence_length=self.sentenceLengths,
                                                                          inputs=self.embeddingInput,
                                                                          dtype=tf.float32)
            BiLSTMOutput = tf.concat(bidOutput, axis=-1)
            self.BiLSTMOutput = tf.nn.dropout(BiLSTMOutput, rate=1 - self.dropout_keep_prob)

    def __addBiLSTMOutPutDenseLayer(self, mode):
        with tf.name_scope('BiLSTMOutputDenseLayer'):
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda)  # 获取正则项
            self.bilstmDenseOutput = keras.layers.Dense(units=self.numClasses,
                                                        activation=keras.activations.relu,
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                        kernel_regularizer=l2_regularizer)(self.BiLSTMOutput)
            # self.bilstmDenseOutput = tf.layers.dense(inputs=self.BiLSTMOutput,
            #                                          units=self.numClasses,
            #                                          activation=tf.nn.relu,
            #                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
            #                                          kernel_regularizer=l2_regularizer)
            self.sequence = tf.argmax(self.bilstmDenseOutput, axis=-1)
            if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.inputY,
                                                                        logits=self.bilstmDenseOutput)
                mask = tf.sequence_mask(self.sentenceLengths)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)
                self.l2_loss = tf.losses.get_regularization_loss()  # 使用get_regularization_loss函数获取定义的全
                self.loss += self.l2_reg_lambda * self.l2_loss

    def __addCrfLayer(self, mode):
        with tf.name_scope('CRFLayer'):
            self.transitionParams = tf.get_variable("transitions", shape=[self.numClasses, self.numClasses],
                                                    initializer=tf.contrib.layers.xavier_initializer())
            self.sequence, _ = crf.crf_decode(self.bilstmDenseOutput,
                                              self.transitionParams,
                                              self.sentenceLengths)
            if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
                logLikelihood, self.transitionParams = crf.crf_log_likelihood(self.bilstmDenseOutput,
                                                                              self.inputY,
                                                                              self.sentenceLengths,
                                                                              transition_params=self.transitionParams)
                self.loss = tf.reduce_mean(-logLikelihood)
                self.loss += self.l2_reg_lambda * self.l2_loss

    def getResult(self, mode):
        self.__addBiLSTMOutPutDenseLayer(mode)
        if self.crf:
            self.__addCrfLayer(mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.sequence
        else:
            weights = tf.sequence_mask(self.sentenceLengths, maxlen=self.maxLength, dtype=tf.int32)
            metrics = {
                'acc': tf.metrics.accuracy(labels=self.inputY, predictions=self.sequence, weights=weights),
                'precision': precision(labels=self.inputY, predictions=self.sequence, num_classes=self.numClasses,
                                       pos_indices=[1, 3, 4, 6, 7, 8], weights=weights),
                'recall': recall(labels=self.inputY, predictions=self.sequence, num_classes=self.numClasses,
                                 pos_indices=[1, 3, 4, 6, 7, 8], weights=weights),
                'f1': f1(labels=self.inputY, predictions=self.sequence, num_classes=self.numClasses,
                         pos_indices=[1, 3, 4, 6, 7, 8], weights=weights)
            }
            if mode == tf.estimator.ModeKeys.TRAIN:
                for metric_name, op in metrics.items():
                    tf.summary.scalar(metric_name, op[1])
                learnRate = tf.train.exponential_decay(self.learnRate, tf.train.get_global_step(), 500, 0.98,
                                                       staircase=True)
                optimizer = tf.train.AdamOptimizer(learnRate)
                self.train_op = optimizer.minimize(self.loss, global_step=tf.train.get_global_step())
                return self.loss, self.train_op
            else:
                return self.loss, metrics
