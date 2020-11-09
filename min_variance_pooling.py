import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from dataset import X_train, X_test, y_train, y_test
from dataset import num_classes, maxlen, num_chars

hdims = 128

class MinVariancePooling(tf.keras.layers.Layer):
	"""最小方差加权平均，Inverse-variance weighting"""

	def __init__(self, **kwargs):
		super(MinVariancePooling, self).__init__(**kwargs)

	def call(self, inputs, mask=None):
		if mask is None:
			mask = 1
		else:
			mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
		mu = tf.reduce_mean(inputs, axis=2, keepdims=True) # 均值
		var = tf.reduce_mean(tf.pow(inputs - mu, 2), axis=2, keepdims=True) # 方差
		var = var - (1 - mask) * 1e12
		ivar = 1 / var
		w = ivar / tf.reduce_sum(ivar, axis=1, keepdims=True)
		return tf.reduce_sum(inputs * w, axis=1)

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
embedding = Embedding(num_chars, hdims, embeddings_initializer="normal", mask_zero=True)

x = embedding(inputs)
x = Conv1D(filters=hdims, kernel_size=2, padding="same", activation="relu")(x)
x = MinVariancePooling()(x, mask=mask)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, shuffle=True, batch_size=32, epochs=10, validation_data=(X_test, y_test))
