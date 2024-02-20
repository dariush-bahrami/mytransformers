from typing import Optional

import torch
from torch import nn

from .multi_head_attention import MultiHeadAttention
from .position_wise_feed_forward import PositionWiseFeedForward


class SequenceTransformerBlock(nn.Module):
    """Transformer Layer. This module consists of a multi-head attention,
    layer normalization and feedforward layer.

    Args:
        queries_embedding_dimension (int): The embedding dimension of the queries.
        keys_embedding_dimension (int): The embedding dimension of the keys.
        values_embedding_dimension (int): The embedding dimension of the values.
        number_of_heads (int): The number of attention heads.
        attention_dropout_p (float): The probability of dropping out a value in the
            attention weights.
        feedforward_dimension (int): The dimension of the intermediate layer. A simple
            rule of thumb is to set this to 4 times the embedding dimension.
        feedforward_dropout_p (float): The probability of dropping out a value in the
            feedforward layer.
    """

    def __init__(
        self,
        queries_embedding_dimension: int,
        keys_embedding_dimension: int,
        values_embedding_dimension: int,
        number_of_heads: int,
        attention_dropout_p: float,
        feedforward_dimension: int,
        feedforward_dropout_p: float,
    ):
        super().__init__()
        self.queries_layer_norm = nn.LayerNorm(queries_embedding_dimension)
        self.keys_layer_norm = nn.LayerNorm(keys_embedding_dimension)
        self.values_layer_norm = nn.LayerNorm(values_embedding_dimension)
        self.feed_forward_layer_norm = nn.LayerNorm(queries_embedding_dimension)
        self.attention = MultiHeadAttention(
            queries_embedding_dimension,
            keys_embedding_dimension,
            values_embedding_dimension,
            queries_embedding_dimension,
            queries_embedding_dimension,
            number_of_heads,
            attention_dropout_p,
        )
        self.feedforward = PositionWiseFeedForward(
            queries_embedding_dimension,
            feedforward_dimension,
            feedforward_dropout_p,
        )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_scale: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the decoder on the embeddings.

        Args:
            queries (torch.Tensor): (B, S1, E1) where E1 is the queries embedding
                dimension.
            keys (torch.Tensor): (B, S2, E2) where E2 is the keys embedding dimension.
            values (torch.Tensor): (B, S2, E3) where E3 is the values embedding
            attention_scale (Tensor, optional): (Broadcastable to (B, S1, S2)). Defaults
                to None. If None, it is set to 1 / sqrt(E1).
            attention_bias (Tensor, optional): (Broadcastable to (B, S1, S2)). Defaults
                to None. This value is added to the scores before the softmax operation.

        Returns:
            torch.Tensor: (B, S1, E1)
        """

        embeddings = (
            self.attention(
                self.queries_layer_norm(queries),
                self.keys_layer_norm(keys),
                self.values_layer_norm(values),
                attention_scale=attention_scale,
                attention_bias=attention_bias,
            )
            + queries
        )
        # Feedforward block
        embeddings = (
            self.feedforward(self.feed_forward_layer_norm(embeddings)) + embeddings
        )
        return embeddings
