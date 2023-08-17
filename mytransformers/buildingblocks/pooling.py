import math

import torch
from torch import nn

from .attention import ScaledDotProductAttention


class FixedLengthWeightedSequencePooling(nn.Module):
    def __init__(self, sequence_length: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(1, sequence_length))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted sequence pooling forward pass.

        Args:
            x (Tensor): (B, S, E)

        Returns:
            Tensor: (B, E)
        """
        return self.weights @ x


class AttentionSequencePooling(nn.Module):
    """Attention sequence pooling module.

    Args:
        embedding_dimension (int): The embedding dimension of the input.
        output_sequence_length (int): The desired length of the output sequence.
        attention_dropout_p (float): The probability of dropping out a value in the
            attention.
    """

    def __init__(
        self,
        embedding_dimension: int,
        output_sequence_length: int,
        attention_dropout_p: float,
    ):
        super().__init__()
        self.query = nn.Parameter(
            torch.empty(output_sequence_length, embedding_dimension)
        )
        nn.init.kaiming_uniform_(self.query, a=math.sqrt(5))
        self.attention = ScaledDotProductAttention(attention_dropout_p)

    def forward(self, x):
        """Attention sequence pooling forward pass.

        Args:
            x (Tensor): (B, S1, E) where S1 = input_sequence_length

        Returns:
            Tensor: (B, S2, E) where S2 = output_sequence_length
        """

        key = value = x  # (B, S1, E)
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)  # (B, S2, E)
        attention_output = self.attention(query, key, value)  # (B, S2, E)
        return attention_output
