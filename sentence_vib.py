import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import datasets
from layers import AttentionPooling1D, VIB, MultiHeadAttentionPooling1D

# imdb: 0.8888

maxlen = 128 * 3
hdims = 128
num_words = 5000
with_vib = False

(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=num_words)
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

inputs = tf.keras.layers.Input(shape=(maxlen,))
x = inputs
x_mask = Lambda(lambda x: tf.not_equal(x, 0))(x)

embedding = Embedding(num_words, output_dim=hdims, embeddings_initializer="normal", mask_zero=True)
conv1 = Conv1D(filters=hdims, kernel_size=2, padding="same", activation="relu")
conv2 = Conv1D(filters=hdims, kernel_size=3, padding="same", activation="relu")
conv3 = Conv1D(filters=hdims, kernel_size=4, padding="same", activation="relu")
# pool = AttentionPooling1D(hdims)
pool = MultiHeadAttentionPooling1D(hdims, heads=2)

x = embedding(x)
x = conv1(x)
x = conv2(x)
x = conv3(x)
x = pool(x, mask=x_mask)

# for VIB
if with_vib:
    d1 = Dense(hdims)
    d2 = Dense(hdims)
    vib = VIB(0.3)

    z_mean = d1(x)
    z_log_var = d2(x)
    x = vib([z_mean, z_log_var])

outputs = Dense(2, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
