import torch
from torch import nn


class PositionWiseFeedForward(nn.Module):
    """Position-wise Feed-forward module.

    Args:
        embedding_dimension (int): The embedding dimension of the input.
        intermediate_dimension (int): The dimension of the intermediate layer.
        dropout_p (float): The probability of dropping out a value in the attention
    """

    def __init__(
        self,
        embedding_dimension: int,
        intermediate_dimension: int,
        dropout_p: float,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.feedforward_dimension = intermediate_dimension
        self.linear_1 = nn.Linear(embedding_dimension, intermediate_dimension)
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(intermediate_dimension, embedding_dimension)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward module to each embedding in the batch and sequence
        separately.

        Args:
            embeddings (torch.Tensor): (B, S, E)

        Returns:
            torch.Tensor: (B, S, E)
        """
        return self.dropout(self.linear_2(self.activation(self.linear_1(embeddings))))
