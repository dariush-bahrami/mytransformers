import math
from typing import Optional, Union

import torch
from torch import nn

# def robust_softmax(x, dim):
#     """Numerically stable softmax."""
#     x_max, _ = x.max(dim=dim, keepdim=True)
#     # replace inf with 0 to avoid overflow
#     x_max.masked_fill_(~torch.isfinite(x_max), 0.0)
#     x = x - x_max
#     x_exp = x.exp()
#     x_exp_sum = x_exp.sum(dim=dim, keepdim=True)
#     # replace 0 with 1 to avoid division by zero
#     x_exp_sum = (x_exp_sum == 0.0) + x_exp_sum
#     return x_exp / x_exp_sum


class DotProductAttention(nn.Module):
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
            query (Tensor): (B, S1, E1)
            key (Tensor): (B, S2, E1)
            value (Tensor): (B, S2, E2)
            scale (Tensor, optional): (Broadcastable to (B, S1, S2)). Defaults to None.
                If None, it is set to 1 / sqrt(E1).
            bias (Tensor, optional): (Broadcastable to (B, S1, S2)). Defaults to None.
                This value is added to the scores before the softmax operation.

        Returns:
            Tensor: (B, S1, E2)
        """
        scores = torch.bmm(query, key.transpose(1, 2))

        if scale is None:
            scale = 1 / math.sqrt(key.shape[-1])
        scores *= scale

        if bias is not None:
            scores += bias

        weights = torch.softmax(scores, dim=-1)
        # replace NaN with 0
        # weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)
        attn_outputs = torch.bmm(weights, value)
        return attn_outputs
