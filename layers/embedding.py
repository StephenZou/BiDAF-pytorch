import tensorflow as tf
from config import max_char_len


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
        self.kernel_size = [2, 3, 4]
        self.conv1d_mapper = {
            size: tf.keras.layers.Conv1D(filters=2, kernel_size=size, strides=1)
            for size in self.kernel_size
        }
        self.pool_mapper = {
            size: tf.keras.layers.MaxPool1D(pool_size=max_char_len-size+1)
            for size in self.kernel_size
        }

    def multi_conv1d(self, x_emb):
        words_emb = tf.unstack(x_emb, axis=1)

    def call(self, cinn_word, qinn_word, cinn_char, qinn_char):
        cemb = self.word_embedding(cinn_word)
        qemb = self.word_embedding(qinn_word)
        c_char_emb = self.char_embedding(cinn_char)
        q_char_emb = self.char_embedding(qinn_char)
        cemb_c =
