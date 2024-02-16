import torch
from torch import nn

from ...building_blocks.attention import PermutedScaledDotProductAttention


class Convention(nn.Module):
    """Self attention based convolutional block.

    The input tensor is first passed through three convolutional layers to obtain the
    query, key and value tensors. Then, these three tensors are passed through self
    attention layer to obtain the attention output.

    Args:
        in_channels (int): Number of channels in the input tensor.
        mid_channels (int): Number of channels in the query and key tensors.
        out_channels (int): Number of channels in the value and output tensors.
        kernel_size (int): Kernel size of the convolutional layers.
        stride (int, optional): Stride of the convolutional layers. Defaults to 1.
        padding (int, optional): Padding of the convolutional layers. Defaults to 0.
        dilation (int, optional): Dilation of the convolutional layers. Defaults to 1.
        groups (int, optional): Groups of the convolutional layers. Defaults to 1.
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
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.query_convolution = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        )
        self.key_convolution = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        )
        self.value_convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        )
        self.flatten = nn.Flatten(start_dim=2)
        self.attention = PermutedScaledDotProductAttention(attention_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        query = self.query_convolution(x)
        output_height = query.shape[2]
        output_width = query.shape[3]
        key = self.key_convolution(x)
        value = self.value_convolution(x)

        query = self.flatten(query)
        key = self.flatten(key)
        value = self.flatten(value)

        attention_output = self.attention(query, key, value)
        output = attention_output.view(
            batch_size,
            self.out_channels,
            output_height,
            output_width,
        )
        return output


class ResidualConvention(nn.Module):
    """Convolutional block with residual connection.

    This module implements the residual connection between the input tensor and the
    output of the convention module. Padding is used to ensure that the input and output
    tensors have the same shape.

    Args:
        in_channels (int): Number of channels in the input tensor.
        mid_channels (int): Number of channels in the query and key tensors.
        out_channels (int): Number of channels in the value and output tensors.
        kernel_size (int): Kernel size of the convolutional layers.
        dilation (int, optional): Dilation of the convolutional layers. Defaults to 1.
        groups (int, optional): Groups of the convolutional layers. Defaults to 1.
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
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = (dilation * (kernel_size - 1)) / 2
        if not padding.is_integer():
            raise Exception(
                "Invalid combination of dilation and kernel_size. "
                + "dilation * (kernel_size - 1) must be even."
            )
        padding = int(padding)
        stride = 1
        super().__init__()
        self.convention = Convention(
            in_channels,
            mid_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
            attention_dropout=attention_dropout,
        )
        self.projection_convolution = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convention(x) + self.projection_convolution(x)


class StandardConventionBlock(nn.Module):
    """Three sequential layers: ResidualConvention > BatchNorm > GELU.

    This module implements the standard convention block which consists of a residual
    convention block followed by a batch normalization and a GELU activation function.

    Args:
        in_channels (int): Number of channels in the input tensor.
        mid_channels (int): Number of channels in the query and key tensors.
        out_channels (int): Number of channels in the value and output tensors.
        kernel_size (int): Kernel size of the convolutional layers.
        dilation (int, optional): Dilation of the convolutional layers. Defaults to 1.
        groups (int, optional): Groups of the convolutional layers. Defaults to 1.
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
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        attention_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ResidualConvention(
                in_channels,
                mid_channels,
                out_channels,
                kernel_size,
                dilation,
                groups,
                bias,
                attention_dropout,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
