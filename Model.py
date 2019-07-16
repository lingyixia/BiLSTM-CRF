import tensorflow as tf
import tensorflow.contrib.crf as crf
from tf_metrics import precision, recall, f1
from tensorflow.contrib import keras


class BiLSTMCrf(object):
    def __init__(self, inputX, inputY, sentenceLengths, numClasses, vocabSize, embeddingSize,
                 hiddenSize, learnRate, maxLength, l2_reg_lambda, dropout_keep_prob, crf, layer_size):
        self.__num_classes = numClasses
        self.__vocab_size = vocabSize
        self.__embedding_size = embeddingSize
        self.__hidden_size = hiddenSize
        self.__learn_rate = learnRate
        self.__max_length = maxLength
        self.l2_reg_lambda = l2_reg_lambda
        self.__inputX = inputX
        self.__inputY = inputY
        self.__sentence_lengths = sentenceLengths
        self.__dropout_keep_prob = dropout_keep_prob
        self.__crf = crf
        self.__layer_size = layer_size
        self.__addEmbeddingLayer()
        self.__addBiLSTMLayer()

    def __addEmbeddingLayer(self):
        with tf.name_scope('embeddingLayer'):
            embedding = tf.get_variable(name='embedding', dtype=tf.float32,
                                        shape=[self.__vocab_size, self.__embedding_size],
                                        initializer=tf.contrib.layers.xavier_initializer())
            embeddingInput = tf.nn.embedding_lookup(embedding, self.__inputX)
            self.embeddingInput = tf.nn.dropout(embeddingInput, rate=1 - self.__dropout_keep_prob)

    def __addBiLSTMLayer(self):
        with tf.name_scope('BiLSTMLayer'):
            bi_layer_size = self.__layer_size // 2
            lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.__hidden_size),
                                               output_keep_prob=self.__dropout_keep_prob) for _ in
                 range(bi_layer_size)])
            lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.__hidden_size),
                                               output_keep_prob=self.__dropout_keep_prob) for _ in
                 range(bi_layer_size)])
            bidOutput, bidCurrent_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                          cell_bw=lstm_bw_cell,
                                                                          sequence_length=self.__sentence_lengths,
                                                                          inputs=self.embeddingInput,
                                                                          dtype=tf.float32)
            BiLSTMOutput = tf.concat(bidOutput, axis=-1)
            self.BiLSTMOutput = tf.nn.dropout(BiLSTMOutput, rate=1 - self.__dropout_keep_prob)

    def __addBiLSTMOutPutDenseLayer(self, mode):
        with tf.name_scope('BiLSTMOutputDenseLayer'):
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale=self.l2_reg_lambda)  # 获取正则项
            self.bilstmDenseOutput = keras.layers.Dense(units=self.__num_classes,
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
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.__inputY,
                                                                        logits=self.bilstmDenseOutput)
                mask = tf.sequence_mask(self.__sentence_lengths)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_mean(losses)
                self.l2_loss = tf.losses.get_regularization_loss()  # 使用get_regularization_loss函数获取定义的全
                self.loss += self.l2_reg_lambda * self.l2_loss

    def __addCrfLayer(self, mode):
        with tf.name_scope('CRFLayer'):
            with tf.variable_scope('transitions', reuse=tf.AUTO_REUSE):
                self.transitionParams = tf.get_variable("transitions", shape=[self.__num_classes, self.__num_classes],
                                                        initializer=tf.contrib.layers.xavier_initializer())
            self.sequence, _ = crf.crf_decode(self.bilstmDenseOutput,
                                              self.transitionParams,
                                              self.__sentence_lengths)
            self.logits = tf.where(tf.not_equal(self.sequence, 0))
            if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
                logLikelihood, self.transitionParams = crf.crf_log_likelihood(self.bilstmDenseOutput,
                                                                              self.__inputY,
                                                                              self.__sentence_lengths)
                self.loss = tf.reduce_mean(-logLikelihood)
                self.loss += self.l2_reg_lambda

    def getResult(self, mode):
        self.__addBiLSTMOutPutDenseLayer(mode)
        if self.__crf:
            self.__addCrfLayer(mode)
        if mode == tf.estimator.ModeKeys.PREDICT:
            return self.sequence
        else:
            weights = tf.sequence_mask(self.__sentence_lengths, dtype=tf.int32)
            metrics = {
                'acc': tf.metrics.accuracy(labels=self.__inputY, predictions=self.sequence, weights=weights),
                'precision': precision(labels=self.__inputY, predictions=self.sequence, num_classes=self.__num_classes,
                                       pos_indices=[0, 1, 2, 4, 6, 7, 8, 9, 10], weights=weights),
                'recall': recall(labels=self.__inputY, predictions=self.sequence, num_classes=self.__num_classes,
                                 pos_indices=[0, 1, 2, 4, 6, 7, 8, 9, 10], weights=weights),
                'f1': f1(labels=self.__inputY, predictions=self.sequence, num_classes=self.__num_classes,
                         pos_indices=[0, 1, 2, 4, 6, 7, 8, 9, 10], weights=weights)
            }
            if mode == tf.estimator.ModeKeys.TRAIN:
                for metric_name, op in metrics.items():
                    tf.summary.scalar(metric_name, op[1])
                learnRate = tf.train.exponential_decay(self.__learn_rate,
                                                       tf.train.get_global_step(),
                                                       500,
                                                       0.98,
                                                       staircase=True)
                optimizer = tf.train.AdamOptimizer(learnRate)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                grads_and_vars_clip = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in grads_and_vars if
                                       grad is not None]
                self.train_op = optimizer.apply_gradients(grads_and_vars_clip, global_step=tf.train.get_global_step())
                return self.loss, self.train_op, self.logits, self.sequence
            else:
                return self.loss, metrics
