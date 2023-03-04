from torch import nn


class TokenEmbedder(nn.Embedding):
    def __init__(self, vocabulary_size: int, embedding_dimension: int):
        super().__init__(vocabulary_size, embedding_dimension)
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
