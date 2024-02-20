from typing import Optional

import torch
from torch import nn

from .dot_product_attention import DotProductAttention


class AttentionHead(nn.Module):
    """Single attention head. This is used in the MultiHeadAttention module.

    Args:
        query_embedding_dimension (int): The embedding dimension of the query.
        key_embedding_dimension (int): The embedding dimension of the key.
        value_embedding_dimension (int): The embedding dimension of the value.
        query_and_key_projection_dimension (int): The projection dimension of the query
            and key.
        value_projection_dimension (int): The projection dimension of the value.
        dropout_p (float): The probability of dropping out a value in the attention
            weights.
    """

    def __init__(
        self,
        query_embedding_dimension: int,
        key_embedding_dimension: int,
        value_embedding_dimension: int,
        query_and_key_projection_dimension: int,
        value_projection_dimension: int,
        dropout_p: float,
    ):
        super().__init__()
        self.scaled_dot_product_attention = DotProductAttention(dropout_p)
        self.query_embedding_dimension = query_embedding_dimension
        self.key_embedding_dimension = key_embedding_dimension
        self.value_embedding_dimension = value_embedding_dimension
        self.query_and_key_projection_dimension = query_and_key_projection_dimension
        self.value_projection_dimension = value_projection_dimension

        self.query_projector = nn.Linear(
            query_embedding_dimension, query_and_key_projection_dimension
        )
        self.key_projector = nn.Linear(
            key_embedding_dimension, query_and_key_projection_dimension
        )
        self.value_projector = nn.Linear(
            value_embedding_dimension, value_projection_dimension
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_scale: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single attention head. This module frist performs a linear projection on the
        query, key, and value tensors based on the embedding dimensions and projection
        dimensions. Then, it performs the scaled dot product attention. Because of the
        projection operations query, key, and value can have different embedding
        dimensions. Other shape constraints are the same as the
        DotProductAttention.

        Args:
            query (Tensor): (B, S1, E1)
            key (Tensor): (B, S2, E2)
            value (Tensor): (B, S2, E3)
            attention_scale (Tensor, optional): (Broadcastable to (B, S1, S2)). Defaults
                to None. If None, it is set to 1 / sqrt(E1).
            attention_bias (Tensor, optional): (Broadcastable to (B, S1, S2)). Defaults
                to None. This value is added to the scores before the softmax operation.

        Returns:
            Tensor: (B, S1, E)
        """
        projected_query = self.query_projector(query)
        projected_key = self.key_projector(key)
        projected_value = self.value_projector(value)
        return self.scaled_dot_product_attention(
            projected_query,
            projected_key,
            projected_value,
            scale=attention_scale,
            bias=attention_bias,
        )
