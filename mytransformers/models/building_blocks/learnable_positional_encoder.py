import torch
from torch import nn


class LearnablePositionalEncoder(nn.Module):
    def __init__(self, embedding_dimension: int, max_sequence_length: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.max_sequence_length = max_sequence_length
        self.positional_encoder = nn.Embedding(max_sequence_length, embedding_dimension)

    def forward(self, embeddings: torch.Tensor):
        device = embeddings.device
        batch_size, sequence_length, embedding_dimension = embeddings.shape
        positions = torch.arange(
            sequence_length, dtype=torch.long, device=device
        ).unsqueeze(0)
        embeddings = embeddings + self.positional_encoder(positions)
        return embeddings
