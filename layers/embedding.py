import tensorflow as tf


class Embedding(tf.keras.layers.Layer):
    def __init__(self, glove_weight,
                 char_vocab_size=0):
        self.glove_weight = glove_weight
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = glove_weight.shape[0]
        self.emb_size = glove_weight.shape[1]
        self.word_embedding = tf.keras.layers.Embedding(self.word_vocab_size, self.emb_size,
                                                        weights=[glove_weight], trainable=False)
        self.char_embedding = tf.keras.layers.Embedding(self.char_vocab_size, self.emb_size,
                                                        embeddings_initializer='uniform')

