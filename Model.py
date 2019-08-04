import tensorflow as tf
import tensorflow.contrib.crf as crf
from tf_metrics import precision, recall, f1
from tensorflow.contrib import keras
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


class BiLSTMCrf(object):
    def __init__(self, inputX, inputY, sentenceLengths, numClasses, vocabSize, embeddingSize,
                 hiddenSize, base_learn_rate, maxLength, l2_reg_lambda, dropout_keep_prob, crf, layer_size, eval_tags,
                 random_seed):
        self.__num_classes = numClasses
        self.__vocab_size = vocabSize
        self.__embedding_size = embeddingSize
        self.__hidden_size = hiddenSize
        self.__base_learn_rate = base_learn_rate
        self.__max_length = maxLength
        self.l2_reg_lambda = l2_reg_lambda
        self.__inputX = inputX
        self.__inputY = inputY
        self.__sentence_lengths = sentenceLengths
        self.__dropout_keep_prob = dropout_keep_prob
        self.__crf = crf
        self.__layer_size = layer_size
        self.__eval_tags = eval_tags
        self.random_seed = random_seed
        self.__addEmbeddingLayer()
        self.__addBiLSTMLayer()

    def __addEmbeddingLayer(self):
        with tf.name_scope('embeddingLayer'):
            embedding = tf.get_variable(name='embedding', dtype=tf.float32,
                                        shape=[self.__vocab_size, self.__embedding_size],
                                        initializer=tf.contrib.layers.xavier_initializer(seed=self.random_seed))
            embeddingInput = tf.nn.embedding_lookup(embedding, self.__inputX)
            self.embeddingInput = tf.nn.dropout(embeddingInput, seed=self.random_seed,
                                                rate=1 - self.__dropout_keep_prob)

    def __addBiLSTMLayer(self):
        with tf.name_scope('BiLSTMLayer'):
            bi_layer_size = self.__layer_size // 2
            lstm_fw_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.__hidden_size),
                                               output_keep_prob=self.__dropout_keep_prob, seed=self.random_seed) for _
                 in
                 range(bi_layer_size)])
            lstm_bw_cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.DropoutWrapper(cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=self.__hidden_size),
                                               output_keep_prob=self.__dropout_keep_prob, seed=self.random_seed) for _
                 in
                 range(bi_layer_size)])
            bidOutput, bidCurrent_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell,
                                                                          cell_bw=lstm_bw_cell,
                                                                          sequence_length=self.__sentence_lengths,
                                                                          inputs=self.embeddingInput,
                                                                          dtype=tf.float32)
            BiLSTMOutput = tf.concat(bidOutput, axis=-1)
            self.BiLSTMOutput = tf.nn.dropout(BiLSTMOutput, seed=self.random_seed, rate=1 - self.__dropout_keep_prob)

    def __addBiLSTMOutPutDenseLayer(self, mode):
        with tf.name_scope('BiLSTMOutputDenseLayer'):
            self.bilstmDenseOutput = keras.layers.Dense(units=self.__num_classes,
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(
                                                            seed=self.random_seed))(self.BiLSTMOutput)
            # self.bilstmDenseOutput = tf.layers.batch_normalization(bilstmDenseOutput,
            #                                                        training=(mode == tf.estimator.ModeKeys.TRAIN),
            #                                                        momentum=0.9)
            self.sequence = tf.argmax(self.bilstmDenseOutput, axis=-1)
            if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.__inputY,
                                                                        logits=self.bilstmDenseOutput)
                mask = tf.sequence_mask(self.__sentence_lengths)
                losses = tf.boolean_mask(losses, mask)
                var_lists = tf.trainable_variables()
                l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                                                                 var_lists)
                self.loss = losses + l2_loss

    def __addCrfLayer(self, mode):
        with tf.name_scope('CRFLayer'):
            with tf.variable_scope('transitions', reuse=tf.AUTO_REUSE):
                self.transitionParams = tf.get_variable("transitions",
                                                        shape=[self.__num_classes, self.__num_classes],
                                                        initializer=tf.contrib.layers.xavier_initializer(
                                                            seed=self.random_seed))
            self.sequence, _ = crf.crf_decode(self.bilstmDenseOutput,
                                              self.transitionParams,
                                              self.__sentence_lengths)
            if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
                logLikelihood, self.transitionParams = crf.crf_log_likelihood(self.bilstmDenseOutput,
                                                                              self.__inputY,
                                                                              self.__sentence_lengths,
                                                                              self.transitionParams)
                self.loss = tf.reduce_mean(-logLikelihood)
                var_lists = tf.trainable_variables()
                l2_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(self.l2_reg_lambda),
                                                                 var_lists)
                self.loss += l2_loss

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
                                       pos_indices=self.__eval_tags, weights=weights),
                'recall': recall(labels=self.__inputY, predictions=self.sequence, num_classes=self.__num_classes,
                                 pos_indices=self.__eval_tags, weights=weights),
                'f1': f1(labels=self.__inputY, predictions=self.sequence, num_classes=self.__num_classes,
                         pos_indices=self.__eval_tags, weights=weights)
            }
            for metric_name, op in metrics.items():
                tf.summary.scalar(metric_name, op[1], family=tf.estimator.ModeKeys.TRAIN)
            if mode == tf.estimator.ModeKeys.TRAIN:
                learn_rate = tf.train.exponential_decay(self.__base_learn_rate,
                                                        tf.train.get_global_step(),
                                                        500,
                                                        0.98,
                                                        staircase=True)
                optimizer = tf.train.AdamOptimizer(learn_rate)
                grads_and_vars = optimizer.compute_gradients(self.loss)
                grads_and_vars_clip = [(tf.clip_by_value(grad, -3, 3), var) for grad, var in grads_and_vars if
                                       grad is not None]
                self.train_op = optimizer.apply_gradients(grads_and_vars_clip, global_step=tf.train.get_global_step())
                return self.loss, self.train_op, learn_rate
            else:
                return self.loss, metrics, self.sequence
