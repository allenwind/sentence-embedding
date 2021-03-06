import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from dataset import X_train, X_test, y_train, y_test
from dataset import num_classes, maxlen, num_chars

hdims = 128

# 0.9105

class MaskGlobalMaxMinPooling1D(tf.keras.layers.Layer):
    """一般来说CNN后加relu激活下，minpooling不起作用，
    这里的实现仅作为实验"""
    
    def __init__(self, **kwargs):
        super(MaskGlobalMaxMinPooling1D, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1
        else:
            # 扩展维度便于广播
            mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
        x = inputs
        x1 = x - (1 - mask) * 1e12 # 用一个大的负数mask
        x1 = tf.reduce_max(x1, axis=1)
        x2 = x + (1 - mask) * 1e12
        x2 = tf.reduce_min(x2, axis=1)
        return tf.concat([x1, x2], axis=1)

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
embedding = Embedding(num_chars, hdims, embeddings_initializer="normal", mask_zero=True)

x = embedding(inputs)
x = Conv1D(filters=hdims, kernel_size=2, padding="same", activation="relu")(x)
x = MaskGlobalMaxMinPooling1D()(x, mask=mask)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, shuffle=True, batch_size=32, epochs=10, validation_data=(X_test, y_test))
