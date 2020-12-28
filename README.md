
## 中心损失函数CenterLoss的实现

在Keras中可通过继承`Layer`的类自定义一个中心损失函数层，在这个新建的层中只需要重写`build`,`call`,`compute_output_shape`函数，其中在`build`中添加一个centers的权重，用于存放中心点，其大小为`num_class x num_features`,即模型输出数x特征数。由于centers不是由梯度来学习的，所以需要设置`trainable=False`。
~~~py

class CenterLossLayer(Layer):
    def __init__(self, num_class, num_features, alpha=0.5, **kwargs):
        # num_class为模型总的分类数量
        # num_features为模型倒数第二层的节点数，即为特征点数
        # alpha为centers的学习率，避免个别标注错误的点对中心点的较大扰动
        super().__init__(**kwargs)
        self.alpha = alpha
        self.num_class = num_class
        self.num_features = num_features

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.num_class, self.num_features),
                                       initializer='uniform', trainable=False)
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
~~~
由于在CenterLossLayer.call中已经返回了centerloss，所以在这部分的损失函数中直接返回中心损失层的输出即可。即需要定义一个损失函数为：
~~~py
def zero_loss(y_true, y_pred):
    return y_pred
~~~

#### issue
如果需要获取centers,按理说只需要`model.get_layer('centerlosslayer').getweights()[0]`即可，但是会报一个需要feed a value for placeholder 'input_2'的错误，找了很久没有找到为什么。 

#### 参考
1. [A Discriminative Feature Learning Approach for Deep Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf)
2. [handongfeng/MNIST-center-loss](https://github.com/handongfeng/MNIST-center-loss/blob/master/centerLoss_MNIST.py)
3. [在Keras使用center-losss损失函数\Keras自定义损失函数](https://blog.csdn.net/DumpDoctorWang/article/details/84204476)
