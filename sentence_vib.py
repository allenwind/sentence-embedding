import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from dataset import X_train, X_test, y_train, y_test
from dataset import num_classes, maxlen, num_chars

hdims = 128

# 0.9096

class MaskGlobalMaxPooling1D(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(MaskGlobalMaxPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x = inputs
        x = x - (1 - mask) * 1e12 # 用一个大的负数mask
        return tf.reduce_max(x, axis=1)

class VIB(tf.keras.layers.Layer):
    """作为句向量的正则化手段"""

    def __init__(self, alpha, **kwargs):
        super(VIB, self).__init__(**kwargs)
        self.alpha = alpha

    def call(self, inputs, training):
        z_mean, z_log_var = inputs
        kl_loss = 0.5 * tf.reduce_sum(
            tf.reduce_mean(tf.square(z_mean) + tf.exp(z_log_var) - 1 - z_log_var, 0)
        )
        self.add_loss(self.alpha * kl_loss)
        if training:
            u_mean = tf.random.normal(shape=tf.shape(z_mean))
        else:
            u_mean = 0.0
        return z_mean + tf.exp(z_log_var / 2) * u_mean

    def compute_output_shape(self, input_shape):
        return input_shape[0]

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
embedding = Embedding(num_chars, hdims, embeddings_initializer="normal", mask_zero=True)

x = embedding(inputs)
x = Conv1D(filters=hdims, kernel_size=2, padding="same", activation="relu")(x)
x = MaskGlobalMaxPooling1D()(x, mask=mask)

# VIB
d1 = Dense(hdims)
d2 = Dense(hdims)
vib = VIB(alpha=0.1)

z_mean = d1(x)
z_log_var = d2(x) # 求log不用激活处理
x = vib([z_mean, z_log_var])

outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, shuffle=True, batch_size=32, epochs=10, validation_data=(X_test, y_test))
