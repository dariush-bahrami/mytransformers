import torch
from torch import nn


class ScaledDotProductAttention(nn.Module):
    """Scaled dot product attention. Identical to the one in the original paper.

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
        attention_mask: torch.Tensor = None,
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
        embed_dim = query.size(-1)
        scores = torch.bmm(query, key.transpose(1, 2)) / (embed_dim**0.5)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == False, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn_outputs = torch.bmm(weights, value)
        return attn_outputs


class AttentionHead(nn.Module):
    """Single attention head. This is used in the MultiHeadAttention module.

    Args:
        embedding_dimension (int): The embedding dimension of the input.
        head_dimension (int): The dimension of the attention head.
        dropout_p (float): The probability of dropping out a value in the attention
            weights. This is the same as the dropout_p in the ScaledDotProductAttention
            module.
    """

    def __init__(self, embedding_dimension: int, head_dimension: int, dropout_p: float):
        super().__init__()
        self.scaled_dot_product_attention = ScaledDotProductAttention(dropout_p)
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.query_projector = nn.Linear(embedding_dimension, head_dimension)
        self.key_projector = nn.Linear(embedding_dimension, head_dimension)
        self.value_projector = nn.Linear(embedding_dimension, head_dimension)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Single attention head. For more details, see the docstrings of the
        ScaledDotProductAttention.

        Args:
            query (Tensor): (B, S1, E)
            key (Tensor): (B, S2, E)
            value (Tensor): (B, S2, E)
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
        embedding_dimension (int): The embedding dimension of the input.
        number_of_heads (int): The number of attention heads.
        dropout_p (float): The probability of dropping out a value in the attention
            weights.
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        dropout_p: float,
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.head_dimension = embedding_dimension // number_of_heads
        self.heads = nn.ModuleList(
            [
                AttentionHead(embedding_dimension, self.head_dimension, dropout_p)
                for _ in range(number_of_heads)
            ]
        )
        self.output_projector = nn.Linear(
            self.head_dimension * number_of_heads, embedding_dimension
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Multi-head attention. For more details, see the docstrings of the
        ScaledDotProductAttention.

        Args:
            query (Tensor): (B, S1, E)
            key (Tensor): (B, S2, E)
            value (Tensor): (B, S2, E)
            attention_mask (Tensor, optional): (B, S1, S2). Defaults to None.

        Returns:
            Tensor: (B, S1, E)
        """
        attn_outputs = torch.cat(
            [head(query, key, value, attention_mask) for head in self.heads], dim=-1
        )
        return self.output_projector(attn_outputs)
