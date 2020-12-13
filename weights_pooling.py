import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from dataset import X_train, X_test, y_train, y_test
from dataset import num_classes, maxlen, num_chars

hdims = 128

class AutoWeightPooling1D(tf.keras.layers.Layer):

    def __init__(self, h_dim, kernel_initializer="glorot_uniform", **kwargs):
        super(AutoWeightPooling1D, self).__init__(**kwargs)
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
            kernel_regularizer="l1", # 添加稀疏性
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
        # 加权平均
        x = tf.reduce_sum(x * x0 * mask, axis=1) / tf.reduce_sum(x * mask, axis=1)
        return x

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
embedding = Embedding(num_chars, hdims, embeddings_initializer="normal", mask_zero=True)

x = embedding(inputs)
x = Conv1D(filters=hdims, kernel_size=2, padding="same", activation="relu")(x)
x = AutoWeightPooling1D(hdims)(x, mask=mask)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, shuffle=True, batch_size=32, epochs=30, validation_data=(X_test, y_test))
