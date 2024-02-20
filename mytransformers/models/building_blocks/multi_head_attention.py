from typing import Optional

import torch
from torch import nn

from .attention_head import AttentionHead


class MultiHeadAttention(nn.Module):
    """Multi-head attention.

    Args:
        query_embedding_dimension (int): The embedding dimension of the query.
        key_embedding_dimension (int): The embedding dimension of the key.
        value_embedding_dimension (int): The embedding dimension of the value.
        query_and_key_projection_dimension (int): The projection dimension of the query
            and key.
        value_projection_dimension (int): The projection dimension of the value.
        number_of_heads (int): The number of attention heads.
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
        number_of_heads: int,
        dropout_p: float,
    ):
        super().__init__()
        self.query_embedding_dimension = query_embedding_dimension
        self.key_embedding_dimension = key_embedding_dimension
        self.value_embedding_dimension = value_embedding_dimension
        self.query_and_key_projection_dimension = query_and_key_projection_dimension
        self.value_projection_dimension = value_projection_dimension
        self.number_of_heads = number_of_heads
        self.dropout_p = dropout_p
        self.each_head_query_and_key_projection_dimension = (
            query_and_key_projection_dimension // number_of_heads
        )
        self.each_head_value_projection_dimension = (
            value_projection_dimension // number_of_heads
        )
        self.heads = nn.ModuleList()
        for _ in range(number_of_heads):
            self.heads.append(
                AttentionHead(
                    query_embedding_dimension,
                    key_embedding_dimension,
                    value_embedding_dimension,
                    self.each_head_query_and_key_projection_dimension,
                    self.each_head_value_projection_dimension,
                    dropout_p,
                )
            )
        self.output_projector = nn.Linear(
            self.each_head_value_projection_dimension * number_of_heads,
            value_projection_dimension,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-head attention. This module pass the query, key, and value tensors
        through multiple attention heads. The output of each attention head is then
        concatenated and passed through a linear layer.


        Args:
            query (Tensor): (B, S1, E1)
            key (Tensor): (B, S2, E2)
            value (Tensor): (B, S2, E3)
            attention_bias (Tensor, optional): (S1, S2) or (B, S1, S2). Defaults to
                None. This value is added to the scores before the softmax operation.

        Returns:
            Tensor: (B, S1, E4) where E4 refers to the value projection dimension.
        """
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(query, key, value, attention_bias=attention_bias))
        concatenated_head_outputs = torch.cat(head_outputs, dim=-1)
        return self.output_projector(concatenated_head_outputs)
