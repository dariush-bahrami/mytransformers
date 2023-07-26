import torch
from torch import nn

from ...buildingblocks.attention import PermutedScaledDotProductAttention


class SelfAttentionFixedResize(nn.Module):
    """Resize the input tensor using self attention to a fixed size.

    This module implements the self attention based resize layer. The idea is similar to
    the ClassificationHead module. A query tensor is created as a learnable parameter.
    The query tensor ensures that the attention output isin (C, H, W) shape where H and
    W are the height and width specified by the user.

    Args:
        in_channels (int): Number of channels in the input tensor.
        mid_channels (int): Number of channels in the query and key tensors.
        out_channels (int): Number of channels in the value and output tensors.
        output_height (int): Height of the output tensor.
        output_width (int): Width of the output tensor.
        bias (bool, optional): Whether to use bias in the convolutional layers.
            Defaults to False.
        attention_dropout (float, optional): Dropout probability of the attention
            layer. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        output_height: int,
        output_width: int,
        bias: bool = False,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_height = output_height
        self.output_width = output_width
        self.query = nn.Parameter(
            torch.randn(
                1,
                mid_channels,
                output_height * output_width,
                requires_grad=True,
            )
        )
        self.key_convolution = nn.Conv2d(
            in_channels,
            mid_channels,
            1,
            1,
            0,
            bias=bias,
        )
        self.value_convolution = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            1,
            0,
            bias=bias,
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.attention = PermutedScaledDotProductAttention(attention_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        key = self.key_convolution(x)
        value = self.value_convolution(x)
        query = self.query.expand(batch_size, -1, -1)
        key = self.flatten(key)
        value = self.flatten(value)
        attention_output = self.attention(query, key, value)
        output = attention_output.view(
            batch_size,
            self.out_channels,
            self.output_height,
            self.output_width,
        )
        return output


class SelfAttentionResize(nn.Module):
    """Resize the input tensor using self attention.

    The difference between SelfAttentionFixedResize and SelfAttentionResize is that the
    height and width of the output tensor can be different for different inputs. If the
    height and width is specified and they can't change for different inputs, then
    SelfAttentionFixedResize may be a better choice because it has more learanble
    parameters in the query tensor.

    Args:
        in_channels (int): Number of channels in the input tensor.
        mid_channels (int): Number of channels in the query and key tensors.
        out_channels (int): Number of channels in the value and output tensors.
        bias (bool, optional): Whether to use bias in the convolutional layers.
            Defaults to False.
        attention_dropout (float, optional): Dropout probability of the attention
            layer. Defaults to 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        bias: bool = False,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.query = nn.Parameter(
            torch.randn(
                1,
                mid_channels,
                1,
                requires_grad=True,
            )
        )
        self.key_convolution = nn.Conv2d(
            in_channels,
            mid_channels,
            1,
            1,
            0,
            bias=bias,
        )
        self.value_convolution = nn.Conv2d(
            in_channels,
            out_channels,
            1,
            1,
            0,
            bias=bias,
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.attention = PermutedScaledDotProductAttention(attention_dropout)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Resize the input tensor to given height and width.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            height (int): Height of the output tensor.
            width (int): Width of the output tensor.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, height, width).
        """

        batch_size = x.shape[0]
        key = self.key_convolution(x)
        value = self.value_convolution(x)
        query = self.query.expand(batch_size, -1, height * width)
        key = self.flatten(key)
        value = self.flatten(value)
        attention_output = self.attention(query, key, value)
        output = attention_output.view(
            batch_size,
            self.out_channels,
            height,
            width,
        )
        return output
