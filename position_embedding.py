import tensorflow as tf

class SinCosInitializer(tf.keras.initializers.Initializer):

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, shape, dtype=None):
        # 使用 numpy 初始化位置向量矩阵
        # embeddings.shape = (1, input_dim, output_dim)
        _, input_dim, output_dim = shape
        pos = np.arange(input_dim)[:, np.newaxis]
        i = np.arange(output_dim)[np.newaxis, :]
        angles = pos / np.power(10000, 2 * i * self.alpha / output_dim)
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        embeddings = tf.cast(angles[np.newaxis, ...], dtype)
        return embeddings

class SinCosPositionEmbedding(tf.keras.layers.Layer):
    """SinCos位置编码"""

    def __init__(self, input_dim=50, output_dim=512, alpha=1.0, trainable=False, **kwargs):
        super(SinCosPositionEmbedding, self).__init__(**kwargs)  
        self.input_dim = input_dim # seq_len
        self.output_dim = output_dim # seq_dim
        self.alpha = alpha # 缩放因子
        self.trainable = trainable

    def build(self, input_shape):
        # embeddings.shape = (1, input_dim, output_dim)
        self.embeddings = self.add_weight(
            name="SinCosPositionEmbedding",
            shape=(1, self.input_dim, self.output_dim),
            initializer=SinCosInitializer(self.alpha),
            trainable=self.trainable
        )

    def call(self, inputs):
        # 根据输入的序列长度返回相应的位置编码
        seq_len = tf.shape(inputs)[1]
        return self.embeddings[:, :seq_len, :]

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.output_dim)

class LearnablePositionEmbedding(tf.keras.layers.Layer):
    """可学习的位置编码"""

    def __init__(
        self,
        input_dim=50,
        output_dim=512,
        embeddings_initializer="zeros",
        **kwargs
    ):
        super(LearnablePositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = tf.keras.initializers.get(
            embeddings_initializer)

    def build(self, input_shape):
        # embeddings.shape = (1, input_dim, output_dim)
        self.embeddings = self.add_weight(
            name="LearnablePositionEmbedding", 
            shape=(1, self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer
        )

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return self.embeddings[:, :seq_len, :]

    def compute_output_shape(self, input_shape):
        return (None, input_shape[1], self.output_dim)

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    a = np.random.randn(1, 150, 510)
    embeddings = SinCosPositionEmbedding(150, alpha=0.3)
    plt.imshow(embeddings(a)[0])
    plt.show()