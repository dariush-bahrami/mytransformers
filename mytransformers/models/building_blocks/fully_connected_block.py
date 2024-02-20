import torch
from torch import nn


class FullyConnectedBlock(nn.Module):
    """Fully connected block. This block applies a linear transformation followed
    by a layer normalization, a GELU activation and a dropout.
    """

    def __init__(self, in_features: int, out_features: int, dropout_p: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.normalizer = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fully connected block forward pass.

        Args:
            x (Tensor): (B, E)

        Returns:
            Tensor: (B, E)
        """
        x = self.linear(x)
        x = self.normalizer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x
