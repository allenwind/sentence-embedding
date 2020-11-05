import tensorflow as tf

class VIB(tf.keras.layers.Layer):

    def __init__(self, l=0.1, **kwargs):
        super(VIB, self).__init__(**kwargs)
        self.l = l

    def call(self, inputs, training):
        z_mean, z_log_var = inputs
        kl_loss = - 0.5 * tf.reduce_sum(tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), 0))
        self.add_loss(self.l * kl_loss)
        if training:
            u = tf.random.normal(shape=tf.shape(z_mean))
        else:
            # u = tf.keras.backend.in_train_phase(u, 0.)
            u = 0.0
        return z_mean + tf.exp(z_log_var / 2) * u

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class AttentionPooling1D(tf.keras.layers.Layer):

    def __init__(self, h_dim, kernel_initializer="glorot_uniform", **kwargs):
        super(AttentionPooling1D, self).__init__(**kwargs)
        self.h_dim = h_dim
        self.kernel_initializer = tf.keras.initializers.get(
            kernel_initializer
        )
        # time steps dim change
        self.supports_masking = False

    def build(self, input_shape):
        self.k_dense = tf.keras.layers.Dense(
            units=self.h_dim,
            kernel_initializer=self.kernel_initializer,
            activation="tanh",
            use_bias=False,
        )
        self.o_dense = tf.keras.layers.Dense(
            units=1,
            kernel_initializer="glorot_uniform",
            use_bias=False
        )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x0 = inputs
        # 计算每个 time steps 权重
        x = self.k_dense(inputs)
        x = self.o_dense(x)
        # 处理 mask
        x = x - (1 - mask) * 1e12
        # 权重归一化
        x = tf.math.softmax(x, axis=1) # 有mask位置对应的权重变为很小的值
        # 加权平均
        x = tf.reduce_sum(x * x0, axis=1)
        
        # 添加重构约束
        return x

class AttentionPooling1DVIB(tf.keras.layers.Layer):

    def __init__(self, h_dim, kernel_initializer="glorot_uniform", **kwargs):
        super(AttentionPooling1DVIB, self).__init__(**kwargs)
        self.h_dim = h_dim
        self.kernel_initializer = tf.keras.initializers.get(
            kernel_initializer
        )
        # time steps dim change
        self.supports_masking = False

    def build(self, input_shape):
        self.k_dense = tf.keras.layers.Dense(
            units=self.h_dim,
            kernel_initializer=self.kernel_initializer,
            activation="tanh",
            use_bias=False,
        )
        self.o_dense = tf.keras.layers.Dense(
            units=1,
            use_bias=False
        )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x0 = inputs
        # 计算每个 time steps 权重
        x = self.k_dense(inputs)
        x = self.o_dense(x)
        # 处理 mask
        x = x - (1 - mask) * 1e12
        # 权重归一化
        x = tf.math.softmax(x, axis=1) # 有mask位置对应的权重变为很小的值
        # 加权平均
        x = tf.reduce_sum(x * x0, axis=1)
        
        # 添加重构约束
        return x


class MultiHeadAttentionPooling1D(tf.keras.layers.Layer):

    def __init__(self, h_dim, heads=8, kernel_initializer="glorot_uniform", **kwargs):
        super(MultiHeadAttentionPooling1D, self).__init__(**kwargs)
        self.h_dim = h_dim
        self.heads = heads
        self.kernel_initializer = tf.keras.initializers.get(
            kernel_initializer
        )
        # time steps dim change
        self.supports_masking = False

    def build(self, input_shape):
        self.k_dense = tf.keras.layers.Dense(
            units=self.h_dim * self.heads,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer="l2",
            activation="tanh",
            use_bias=False,
        )
        # 计算多个注意力分布
        self.o_dense = tf.keras.layers.Dense(
            units=self.heads,
            kernel_initializer="glorot_uniform",
            kernel_regularizer="l2",
            use_bias=False
        )

    def call(self, inputs, mask=None):
        batch_size = tf.shape(inputs)[0]
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x0 = inputs
        # 计算每个 time steps 权重
        x = self.k_dense(inputs)
        x = self.o_dense(x)
        # (batch_size, seq_len, heads, h_dim)
        # x = tf.reshape(x, (batch_size, -1, self.heads, self.h_dim))
        # 处理 mask
        x = x - (1 - mask) * 1e12
        # 权重归一化
        x = tf.math.softmax(x, axis=1) # 有mask位置对应的权重变为很小的值
        x = tf.einsum("blh,bld->bhd", x, x0)
        x = tf.reshape(x, shape=(batch_size, self.h_dim * self.heads))
        return x
