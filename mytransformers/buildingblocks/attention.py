from typing import Optional

import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot product attention.

    Args:
        dropout_p (float): The probability of dropping out a value in the attention
            weights.
    """

    def __init__(self, dropout_p: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scaled dot product attention. All inputs shapes are in the form of (B, S, E)
        where B is the batch size, S is the sequence length, and E is the embedding
        dimension.

        Note About Shapes:
            - The batch dimension of query, key, and value must be the same.
            - The sequence dimension of the key and value must be the same.
            - The embedding dimension of query and key must be the same.

        Args:
            query (Tensor): (B, S1, E1)
            key (Tensor): (B, S2, E1)
            value (Tensor): (B, S2, E2)
            attention_mask (Tensor, optional): (B, S1, S2). Defaults to None. The dtype
                of the mask should be torch.bool. False means that the corresponding
                attention weight should be zeroed out.

        Returns:
            Tensor: (B, S1, E2)
        """
        dk = key.shape[-1]
        scores = torch.bmm(query, key.transpose(1, 2)) / (dk**0.5)
        if attention_mask is not None:
            scores = torch.masked_fill(scores, attention_mask == False, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn_outputs = torch.bmm(weights, value)
        return attn_outputs


class PermutedScaledDotProductAttention(nn.Module):
    """Permuted Scaled dot product attention.

    This is the same as ScaledDotProductAttention except that the embedding dimension
    is the first dimension of the query, key, and value tensors. This is useful for
    convolutional attention.

    Args:
        dropout_p (float): The probability of dropping out a value in the attention
            weights.
    """

    def __init__(self, dropout_p: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scaled dot product attention. All inputs shapes are in the form of (B, S, E)
        where B is the batch size, S is the sequence length, and E is the embedding
        dimension.

        Note About Shapes:
            - The batch dimension of query, key, and value must be the same.
            - The sequence dimension of the key and value must be the same.
            - The embedding dimension of query and key must be the same.

        Args:
            query (Tensor): (B, E1, S1)
            key (Tensor): (B, E1, S2)
            value (Tensor): (B, E2, S2)
            attention_mask (Tensor, optional): (B, S1, S2). Defaults to None. The dtype
                of the mask should be torch.bool. False means that the corresponding
                attention weight should be zeroed out.

        Returns:
            Tensor: (B, E2, S1)
        """
        dk = key.shape[-2]
        scores = torch.bmm(query.transpose(1, 2), key) / (dk**0.5)
        if attention_mask is not None:
            scores = torch.masked_fill(scores, attention_mask == False, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn_outputs = torch.bmm(value, weights.transpose(1, 2))
        return attn_outputs


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
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_p)
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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single attention head. This module frist performs a linear projection on the
        query, key, and value tensors based on the embedding dimensions and projection
        dimensions. Then, it performs the scaled dot product attention. Because of the
        projection operations query, key, and value can have different embedding
        dimensions. Other shape constraints are the same as the
        ScaledDotProductAttention.

        Args:
            query (Tensor): (B, S1, E1)
            key (Tensor): (B, S2, E2)
            value (Tensor): (B, S2, E3)
            attention_mask (Tensor, optional): (B, S1, S2). Defaults to None.

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
            attention_mask,
        )


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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-head attention. This module pass the query, key, and value tensors
        through multiple attention heads. The output of each attention head is then
        concatenated and passed through a linear layer.


        Args:
            query (Tensor): (B, S1, E1)
            key (Tensor): (B, S2, E2)
            value (Tensor): (B, S2, E3)
            attention_mask (Tensor, optional): (B, S1, S2). Defaults to None.

        Returns:
            Tensor: (B, S1, E4) where E4 refers to the value projection dimension.
        """
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(query, key, value, attention_mask))
        concatenated_head_outputs = torch.cat(head_outputs, dim=-1)
        return self.output_projector(concatenated_head_outputs)
