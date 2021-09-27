import torch
import torch.nn as nn
from config import max_char_len


class BiDAF:
    def __init__(
            self, clen, qlen,
            num_highway_layers=2,
            encoder_dropout=0,
            num_decoders=2,
            decoder_dropout=0,
            glove_weight=None,
            char_vocab_size=0
    ):
        self.clen = clen
        self.qlen = qlen
        self.num_highway_layers = num_highway_layers
        self.encoder_dropout = encoder_dropout
        self.num_decoders = num_decoders
        self.decoder_dropout = decoder_dropout
        self.glove_weight = glove_weight
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = 0 if glove_weight is None else glove_weight.shape[0]
        self.emb_size = 0 if glove_weight is None else glove_weight.shape[1]
        self.kernel_size = [2, 3, 4]
        self.conv1d_mapper = {
            size: nn.Conv1d(in_channels=self.emb_size, out_channels=2, kernel_size=size, stride=1)
            for size in self.kernel_size
        }
        self.pool_mapper = {
            size: nn.MaxPool1d(kernel_size=max_char_len - size + 1)
            for size in self.kernel_size
        }

    def build_model(self):
        word_embedding_layer = nn.Embedding(num_embeddings=self.word_vocab_size,
                                            embedding_dim=self.emb_size)
        word_embedding_layer.weight.data.copy_(torch.)
