from keras.callbacks import TensorBoard
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
from keras import losses
from keras.engine.topology import Layer
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.layers.advanced_activations import PReLU
from keras import initializers
from keras import backend as K
import numpy as np
import tensorflow as tf

import my_callbacks

### parameters

initial_learning_rate = 1e-3
batch_size = 4
epochs = 5
weight_decay = 0.0005


### prelu

def prelu(x, name='default'):
    if name == 'default':
        return PReLU(alpha_initializer=initializers.Constant(value=0.25))(x)
    else:
        return PReLU(alpha_initializer=initializers.Constant(value=0.25), name=name)(x)


### special layer

class CenterLossLayer(Layer):

    def __init__(self, alpha=0.5, num_class=10, num_features=2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.num_class = num_class
        self.num_features = num_features

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(10, 2),
                                       initializer='uniform',
                                       trainable=False)
        super().build(input_shape)

    def call(self, x, mask=None):
        """
        计算中心损失
        :param x: 为centerLossLayer层的输入，为2个tensor,
                    x[0]为模型的特征向量(batch_size x num_features)，
                    x[1]为label标签，非onehot(batch_size x 1)
        :param mask: None
        :return:
        """
        # 将x[1]转化为batch_size长度的向量，并转化为int32数据类型

        labels = tf.reshape(x[1], [-1])
        labels = tf.cast(labels, 'int32')
        # 获取当前batch下涉及到可能需要更新的centers
        center_batch = tf.gather(self.centers, labels)
        # 计算当前batch下中心更新量
        diff = center_batch - x[0]
        # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        diff = diff / tf.cast((1 + appear_times), tf.float32)

        # 更新centers: 用减法的形式更新，需要更新的是self.centers, 更新的index是labels表示，更新量为self.alpha * diff
        self.centers = tf.scatter_sub(self.centers, labels, self.alpha * diff)

        # 计算当前batch下的centerloss
        self.result = x[0] - K.dot(tf.one_hot(tf.cast(x[1], 'int32'), self.num_class), self.centers)
        self.result = 0.5 * K.sum(self.result ** 2, axis=1, keepdims=True)
        return self.result

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)


def zero_loss(y_true, y_pred):
    return y_pred


def my_model(x, labels):
    x = BatchNormalization()(x)
    #
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
    #
    x = Flatten()(x)
    x = Dense(2, kernel_regularizer=l2(weight_decay))(x)
    x = prelu(x, name='side_out')
    #
    main = Dense(10, activation='softmax', name='main_out', kernel_regularizer=l2(weight_decay))(x)
    side = CenterLossLayer(alpha=0.5, name='centerlosslayer')([x, labels])
    return main, side


def run(lambda_centerloss):
    """
    Run the model
    :param lambda_centerloss:
    :return:
    """

    ### get data

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_train_onehot = to_categorical(y_train, 10)
    y_test_onehot = to_categorical(y_test, 10)

    ### compile

    main_input = Input((28, 28, 1))
    aux_input = Input((1,))

    final_output, side_output = my_model(main_input, aux_input)

    model = Model(inputs=[main_input, aux_input], outputs=[final_output, side_output])
    model.summary()

    optim = optimizers.SGD(lr=initial_learning_rate, momentum=0.9)
    model.compile(optimizer=optim,
                  loss=[losses.categorical_crossentropy, zero_loss],
                  loss_weights=[1, lambda_centerloss],
                  metrics=['acc'])

    ### fit

    dummy1 = np.zeros((x_train.shape[0], 1))
    dummy2 = np.zeros((x_test.shape[0], 1))

    model.fit([x_train, y_train], [y_train_onehot, dummy1], batch_size=batch_size,
              epochs=epochs,
              verbose=1, validation_data=([x_test, y_test], [y_test_onehot, dummy2]))

    reduced_model = Model(inputs=model.input[0], outputs=model.get_layer('side_out').output)
    reduced_model.summary()

    feats = reduced_model.predict(x_train, verbose=1)
    my_callbacks.visualize_train(feats, y_train, epoch=epochs - 1,
                                 centers=None,
                                 lambda_cl=lambda_centerloss)
    feats = reduced_model.predict(x_test, verbose=1)
    my_callbacks.visualize(feats, y_test, epoch=epochs - 1,
                           centers=None,
                           lambda_cl=lambda_centerloss)


if __name__ == '__main__':
    run(0.1)
