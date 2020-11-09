import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from dataset import X_train, X_test, y_train, y_test
from dataset import num_classes, maxlen, num_chars

hdims = 128

class PowerMeanPooling(tf.keras.layers.Layer):
	"""类似Minkowski distance"""

	def __init__(self, p=2, **kwargs):
		super(PowerMeanPooling, self).__init__(**kwargs)
		self.p = p

	def call(self, inputs, mask=None):
		if mask is None:
			mask = 1
		else:
			mask = tf.expand_dims(tf.cast(mask, "float32"), -1)
		x = tf.pow(inputs, self.p)
		x = tf.reduce_sum(x, axis=1)
		x = x / (tf.reduce_sum(mask, axis=1) + 1e-12)
		return tf.pow(x, 1 / self.p)

inputs = Input(shape=(maxlen,))
mask = Lambda(lambda x: tf.not_equal(x, 0))(inputs)
embedding = Embedding(num_chars, hdims, embeddings_initializer="normal", mask_zero=True)

x = embedding(inputs)
x = Conv1D(filters=hdims, kernel_size=2, padding="same", activation="relu")(x)
x = PowerMeanPooling(p=2)(x, mask=mask)
outputs = Dense(num_classes, activation="softmax")(x)
model = Model(inputs, outputs)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(X_train, y_train, shuffle=True, batch_size=32, epochs=10, validation_data=(X_test, y_test))
