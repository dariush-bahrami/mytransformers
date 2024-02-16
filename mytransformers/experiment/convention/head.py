import torch
from torch import nn

from ...building_blocks.attention import PermutedScaledDotProductAttention


class ClassificationHead(nn.Module):
    """Self attention based classification head.

    This module implements the classification head which uses the the same idea as the
    convention block. The input tensor is used as the key tensor. The input tensor is
    passed through a convolutional layer to obtain the value tensor. A query tensor is
    created as a learnable parameter. the query tensor ensures that the attention output
    is in (C, 1, 1) shape. The output of the attention layer is used as the logits of
    the classification head.

    Args:
        in_channels (int): Number of channels in the input tensor.
        num_classes (int): Number of classes in the classification task.
        attention_dropout (float, optional): Dropout probability of the attention
            layer. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.value_convolution = nn.Conv2d(
            in_channels, num_classes, 1, 1, 0, bias=False
        )
        self.query = nn.Parameter(
            torch.randn(
                1,
                in_channels,
                1,
                requires_grad=True,
            )
        )
        self.attention = PermutedScaledDotProductAttention(attention_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        query = self.query.expand(batch_size, -1, -1)
        key = x
        value = self.value_convolution(x)
        key = key.view(batch_size, self.in_channels, -1)
        value = value.view(batch_size, self.num_classes, -1)
        attention_output = self.attention(query, key, value)
        output = attention_output.view(batch_size, self.num_classes)
        return output
