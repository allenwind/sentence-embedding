import tensorflow as tf

class SimpleEmbedding(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, mask_zero=False, **kwargs):
        super(SimpleEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer="glorot_normal",
            dtype="float32",
        )

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)

if __name__ == "__main__":
    e = SimpleEmbedding(5, 10)
    x = tf.one_hot([0, 1, 3], depth=5, dtype="int32")
    print(e(x))

    for i in range(5):
        print(e([[i]]))
