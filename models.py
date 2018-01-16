"""
Todo:
Try Adagrad and Adadelta
Implement cnn+lstm
Implement FastText embedding + models
Research/implement Region embeddings from http://proceedings.mlr.press/v48/johnson16.pdf
Experiment with Attention and Dilated Convolutions
"""

from keras import regularizers
from keras.activations import relu
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers import RandomNormal
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D, MaxPooling1D, LSTM, Bidirectional, Dense, Dropout, \
    PReLU, BatchNormalization, Lambda
from keras.layers.merge import add
from keras.models import Model
from keras.optimizers import Adam, SGD, Adagrad, Adadelta

from steps.keras.callbacks import NeptuneMonitor, ReduceLR
from steps.keras.models import ClassifierXY
from steps.utils import create_filepath


class CharacterClassifier(ClassifierXY):
    def _build_optimizer(self, **kwargs):
        return Adam(**kwargs)

    def _build_loss(self, **kwargs):
        return 'binary_crossentropy'

    def _create_callbacks(self, **kwargs):
        lr_scheduler = ReduceLR(**kwargs['lr_scheduler'])
        early_stopping = EarlyStopping(**kwargs['early_stopping'])
        checkpoint_filepath = kwargs['model_checkpoint']['filepath']
        create_filepath(checkpoint_filepath)
        model_checkpoint = ModelCheckpoint(**kwargs['model_checkpoint'])
        neptune = NeptuneMonitor()
        return [neptune, lr_scheduler, early_stopping, model_checkpoint]


class CharVDCNN(CharacterClassifier):
    def _build_optimizer(self, **kwargs):
        return Adam(**kwargs)

    def _build_model(self, embedding_size,
                     maxlen, max_features,
                     filter_nr, kernel_size, repeat_block, dropout_convo,
                     dense_size, repeat_dense, dropout_dense,
                     l2_reg_convo, l2_reg_dense, use_prelu, use_batch_norm):
        return vdcnn(embedding_size,
                     maxlen, max_features,
                     filter_nr, kernel_size, repeat_block, dropout_convo,
                     dense_size, repeat_dense, dropout_dense,
                     l2_reg_convo, l2_reg_dense, use_prelu, use_batch_norm)


class WordLSTM(CharacterClassifier):
    def _build_optimizer(self, **kwargs):
        return Adam(**kwargs)

    def _build_model(self, embedding_size,
                     maxlen, max_features,
                     unit_nr, repeat_block, dropout_lstm,
                     dense_size, repeat_dense, dropout_dense,
                     l2_reg_dense, use_prelu, use_batch_norm, global_pooling):
        return lstm(None, embedding_size,
                    maxlen, max_features,
                    unit_nr, repeat_block, dropout_lstm,
                    dense_size, repeat_dense, dropout_dense,
                    l2_reg_dense, use_prelu, use_batch_norm, False, global_pooling)


class WordDPCNN(CharacterClassifier):
    def _build_optimizer(self, **kwargs):
        return Adam(**kwargs)

    def _build_model(self, embedding_size,
                     maxlen, max_features,
                     filter_nr, kernel_size, repeat_block, dropout_convo,
                     dense_size, repeat_dense, dropout_dense,
                     l2_reg_convo, l2_reg_dense, use_prelu, trainable_embedding, use_batch_norm):
        """
        Implementation of http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
        """
        return dpcnn(None, embedding_size,
                     maxlen, max_features,
                     filter_nr, kernel_size, repeat_block, dropout_convo,
                     dense_size, repeat_dense, dropout_dense,
                     l2_reg_convo, l2_reg_dense, use_prelu, trainable_embedding, use_batch_norm)


class GloveBasic(CharacterClassifier):
    def fit(self, embedding_matrix, X, y, validation_data):
        X_valid, y_valid = validation_data
        self.callbacks = self._create_callbacks(**self.callbacks_config)
        self.architecture_config['model_params']['embedding_matrix'] = embedding_matrix
        self.model = self._compile_model(**self.architecture_config)
        self.model.fit(X, y,
                       validation_data=[X_valid, y_valid],
                       callbacks=self.callbacks,
                       verbose=1,
                       **self.training_config)
        return self

    def transform(self, embedding_matrix, X, y=None, validation_data=None):
        predictions = self.model.predict(X, verbose=1)
        return {'prediction_probability': predictions}


class GloveLSTM(GloveBasic):
    def _build_optimizer(self, **kwargs):
        return Adam(**kwargs)

    def _build_model(self, embedding_matrix, embedding_size,
                     maxlen, max_features,
                     unit_nr, repeat_block, dropout_lstm,
                     dense_size, repeat_dense, dropout_dense,
                     l2_reg_dense, use_prelu, use_batch_norm, trainable_embedding, global_pooling):
        return lstm(embedding_matrix, embedding_size,
                    maxlen, max_features,
                    unit_nr, repeat_block, dropout_lstm,
                    dense_size, repeat_dense, dropout_dense,
                    l2_reg_dense, use_prelu, use_batch_norm, trainable_embedding, global_pooling)


class GloveSCNN(GloveBasic):
    def _build_optimizer(self, **kwargs):
        return SGD(**kwargs)

    def _build_model(self, embedding_matrix, embedding_size,
                     maxlen, max_features,
                     filter_nr, kernel_size, dropout_convo,
                     dense_size, repeat_dense, dropout_dense,
                     l2_reg_convo, l2_reg_dense, use_prelu, trainable_embedding, use_batch_norm):
        return scnn(embedding_matrix, embedding_size,
                    maxlen, max_features,
                    filter_nr, kernel_size, dropout_convo,
                    dense_size, repeat_dense, dropout_dense,
                    l2_reg_convo, l2_reg_dense, use_prelu, trainable_embedding, use_batch_norm)


class GloveDPCNN(GloveBasic):
    def _build_optimizer(self, **kwargs):
        return SGD(**kwargs)

    def _build_model(self, embedding_matrix, embedding_size,
                     maxlen, max_features,
                     filter_nr, kernel_size, repeat_block, dropout_convo,
                     dense_size, repeat_dense, dropout_dense,
                     l2_reg_convo, l2_reg_dense, use_prelu, trainable_embedding, use_batch_norm):
        """
        Implementation of http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
        """
        return dpcnn(embedding_matrix, embedding_size,
                     maxlen, max_features,
                     filter_nr, kernel_size, repeat_block, dropout_convo,
                     dense_size, repeat_dense, dropout_dense,
                     l2_reg_convo, l2_reg_dense, use_prelu, trainable_embedding, use_batch_norm)


class GloveCLSTM(GloveBasic):
    def _build_optimizer(self, **kwargs):
        return SGD(**kwargs)

    def _build_model(self, embedding_matrix, embedding_size,
                     maxlen, max_features,
                     filter_nr, kernel_size, repeat_block, dropout_convo,
                     dense_size, repeat_dense, dropout_dense,
                     l2_reg_convo, l2_reg_dense, use_prelu, trainable_embedding, use_batch_norm):
        return clstm(embedding_matrix, embedding_size,
                     maxlen, max_features,
                     filter_nr, kernel_size, repeat_block, dropout_convo,
                     dense_size, repeat_dense, dropout_dense,
                     l2_reg_convo, l2_reg_dense, use_prelu, trainable_embedding, use_batch_norm)


def scnn(embedding_matrix, embedding_size,
         maxlen, max_features,
         filter_nr, kernel_size, dropout_convo,
         dense_size, repeat_dense, dropout_dense,
         l2_reg_convo, l2_reg_dense, use_prelu, trainable_embedding, use_batch_norm):
    input_text = Input(shape=(maxlen,))
    x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=trainable_embedding)(
        input_text)
    x = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout_convo, l2_reg_convo)(x)

    x = GlobalMaxPool1D()(x)
    for _ in range(repeat_dense):
        x = _dense_block(dense_size, use_batch_norm, use_prelu, dropout_dense, l2_reg_dense)(x)
    predictions = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_text, outputs=predictions)
    return model


def dpcnn(embedding_matrix, embedding_size,
          maxlen, max_features,
          filter_nr, kernel_size, repeat_block, dropout_convo,
          dense_size, repeat_dense, dropout_dense,
          l2_reg_convo, l2_reg_dense, use_prelu,
          trainable_embedding, use_batch_norm):

    """
    Note:
        Implementation of http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
        post activation is used instead of pre-activation, could be worth exploring
    """

    input_text = Input(shape=(maxlen,))
    if embedding_matrix is not None:
        embedding = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=trainable_embedding)(
            input_text)
    else:
        embedding = Embedding(max_features, embedding_size)(input_text)
    x = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout_convo, l2_reg_convo)(embedding)
    x = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout_convo, l2_reg_convo)(x)
    if embedding_size == filter_nr:
        x = add([embedding, x])
    else:
        embedding_resized = _shape_matching_layer(filter_nr, use_prelu, dropout_convo, l2_reg_convo)(embedding)
        x = add([embedding_resized, x])
    for _ in range(repeat_block):
        x = _dpcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout_convo, l2_reg_convo)(x)

    x = GlobalMaxPool1D()(x)
    for _ in range(repeat_dense):
        x = _dense_block(dense_size, use_batch_norm, use_prelu, dropout_dense, l2_reg_dense)(x)
    predictions = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=input_text, outputs=predictions)
    return model


def lstm(embedding_matrix, embedding_size,
         maxlen, max_features,
         unit_nr, repeat_block, dropout_lstm,
         dense_size, repeat_dense, dropout_dense,
         l2_reg_dense, use_prelu, use_batch_norm, trainable_embedding, global_pooling):
    input_text = Input(shape=(maxlen,))
    if embedding_matrix is not None:
        x = Embedding(max_features, embedding_size, weights=[embedding_matrix], trainable=trainable_embedding)(
            input_text)
    else:
        x = Embedding(max_features, embedding_size)(input_text)
    for _ in range(repeat_block - 1):
        x = _lstm_block(unit_nr, return_sequences=True, dropout_lstm=dropout_lstm)(x)
    if global_pooling:
        x = _lstm_block(unit_nr, return_sequences=True, dropout_lstm=dropout_lstm)(x)
        x = GlobalMaxPool1D()(x)
    else:
        x = _lstm_block(unit_nr, return_sequences=False, dropout_lstm=dropout_lstm)(x)
    for _ in range(repeat_dense):
        x = _dense_block(dense_size, use_batch_norm, use_prelu, dropout_dense, l2_reg_dense)(x)
    predictions = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=input_text, outputs=predictions)
    return model


def vdcnn(embedding_size,
          maxlen, max_features,
          filter_nr, kernel_size, repeat_block, dropout_convo,
          dense_size, repeat_dense, dropout_dense,
          l2_reg_convo, l2_reg_dense, use_prelu, use_batch_norm):
    """
    Note:
        Implementation of http://www.aclweb.org/anthology/E17-1104
        We didn't use k-max pooling but GlobalMaxPool1D at the end and didn't explore it in the
        intermediate layers.
    """

    input_text = Input(shape=(maxlen,))
    x = Embedding(input_dim=max_features, output_dim=embedding_size)(input_text)
    x = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout_convo, l2_reg_convo)(x)

    for i in range(repeat_block):
        if i + 1 != repeat_block:
            x = _vdcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout_convo, l2_reg_convo,
                             last_block=False)(x)
        else:
            x = _vdcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout_convo, l2_reg_convo,
                             last_block=True)(x)

    x = GlobalMaxPool1D()(x)
    for i in range(repeat_dense):
        x = _dense_block(dense_size, use_batch_norm, use_prelu, dropout_dense, l2_reg_dense)(x)
    predictions = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=input_text, outputs=predictions)
    return model


def clstm(embedding_matrix, embedding_size,
          maxlen, max_features,
          filter_nr, kernel_size, repeat_block, dropout_convo,
          dense_size, repeat_dense, dropout_dense,
          l2_reg_convo, l2_reg_dense, use_prelu, trainable_embedding, use_batch_norm):
    """
    Implementation of https://arxiv.org/pdf/1511.08630.pdf
    """
    return NotImplementedError


def _bn_relu_dropout_block(use_batch_norm, use_prelu, dropout):
    def f(x):
        if use_batch_norm:
            x = BatchNormalization()(x)
        if use_prelu:
            x = PReLU()(x)
        else:
            x = Lambda(relu)(x)
        x = Dropout(dropout)(x)
        return x

    return f


def _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, l2_reg):
    def f(x):
        x = Conv1D(filter_nr, kernel_size=kernel_size, padding='same', activation='linear',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                   kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = _bn_relu_dropout_block(use_batch_norm, use_prelu, dropout)(x)
        return x

    return f


def _shape_matching_layer(filter_nr, use_prelu, dropout, l2_reg):
    def f(x):
        x = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.001),
                   kernel_regularizer=regularizers.l2(l2_reg))(x)
        if use_prelu:
            x = PReLU()(x)
        else:
            x = Lambda(relu)(x)
        x = Dropout(dropout)(x)
        return x

    return f


def _lstm_block(unit_nr, return_sequences, dropout_lstm, bidirectional=True):
    def f(x):
        if bidirectional:
            x = Bidirectional(
                LSTM(unit_nr,
                     return_sequences=return_sequences,
                     dropout=dropout_lstm,
                     recurrent_dropout=dropout_lstm))(x)
        else:
            x = LSTM(unit_nr,
                     return_sequences=return_sequences,
                     dropout=dropout_lstm,
                     recurrent_dropout=dropout_lstm)(x)
        return x

    return f


def _dense_block(dense_size, use_batch_norm, use_prelu, dropout, l2_reg):
    def f(x):
        x = Dense(dense_size, activation='linear',
                  kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = _bn_relu_dropout_block(use_batch_norm, use_prelu, dropout)(x)
        return x

    return f


def _dpcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, l2_reg):
    def f(x):
        x = MaxPooling1D(pool_size=3, strides=2)(x)
        main = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, l2_reg)(x)
        main = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, l2_reg)(main)
        x = add([main, x])
        return x

    return f


def _vdcnn_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, l2_reg, last_block):
    def f(x):
        main = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, l2_reg)(x)
        x = add([main, x])
        main = _convolutional_block(filter_nr, kernel_size, use_batch_norm, use_prelu, dropout, l2_reg)(x)
        x = add([main, x])
        if not last_block:
            x = MaxPooling1D(pool_size=3, strides=2)(x)
        return x

    return f
