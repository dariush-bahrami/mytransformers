import torch
from torch import nn


class SinusoidalPositionalEncoder(nn.Module):
    @staticmethod
    def get_positional_encodings(embedding_dimension: int, sequence_length: int):
        position = torch.arange(sequence_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dimension, 2)
            * (-torch.log(10000.0) / embedding_dimension)
        )
        pe = torch.zeros(sequence_length, 1, embedding_dimension)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return pe

    def __init__(
        self, embedding_dimension: int, max_sequence_length: int, dropout_p: float
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length
        self.dropout = nn.Dropout(dropout_p)
        self.register_buffer(
            "positional_encodings",
            self.get_positional_encodings(embedding_dimension, max_sequence_length),
        )

    def forward(self, embeddings: torch.Tensor):
        batch_size, sequence_length, embedding_dimension = embeddings.shape
        embeddings = embeddings + self.positional_encodings[:sequence_length]
        embeddings = self.dropout(embeddings)
        return embeddings


class LearnablePositionalEncoder(nn.Module):
    def __init__(
        self, embedding_dimension: int, max_sequence_length: int, dropout_p: float
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length
        self.dropout = nn.Dropout(dropout_p)
        self.positional_encoder = nn.Embedding(max_sequence_length, embedding_dimension)

    def forward(self, embeddings: torch.Tensor):
        device = embeddings.device
        batch_size, sequence_length, embedding_dimension = embeddings.shape
        positions = torch.arange(
            sequence_length, dtype=torch.long, device=device
        ).unsqueeze(0)
        embeddings = embeddings + self.positional_encoder(positions)
        embeddings = self.dropout(embeddings)
        return embeddings
