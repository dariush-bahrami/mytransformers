import math
from typing import Optional, Union

import torch
from torch import nn


class PermutedDotProductAttention(nn.Module):
    """Permuted Scaled dot product attention.

    This is the same as DotProductAttention except that the embedding dimension
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
        scale: Optional[Union[float, torch.Tensor]] = None,
        bias: Optional[torch.Tensor] = None,
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
            scale (Tensor, optional): (Broadcastable to (B, S1, S2)). Defaults to None.
                If None, it is set to 1 / sqrt(E1).
            bias (Tensor, optional): (Broadcastable to (B, S1, S2)). Defaults to None.
                This value is added to the scores before the softmax operation.

        Returns:
            Tensor: (B, E2, S1)
        """
        scores = torch.bmm(query.transpose(1, 2), key)

        if scale is None:
            scale = 1 / math.sqrt(key.shape[-1])
        scores *= scale

        if bias is not None:
            scores += bias

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        attn_outputs = torch.bmm(value, weights.transpose(1, 2))
        return attn_outputs
