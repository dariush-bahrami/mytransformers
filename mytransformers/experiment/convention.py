"""Convolutions with Attention.

In this module, A self attention based convolutional block is implemented. The main
idea is to use convolutional layers instead of linear layers in the attention head of
the paper "Attention is all you need." This perspective leads to other ways of
implementing the common operations on images. For example the ClassificationHead,
SelfAttentionFixedResize and SelfAttentionResize are implemented using this mindset.
"""

import torch
from torch import nn

from ..buildingblocks.attention import PermutedScaledDotProductAttention


class PositionalEncoding2D(nn.Module):
    """Concatenate normalized position coordinates to the input tensor.

    This module concatenates the X and Y coordinates of the input tensor as two
    additional channels to it. X and Y coordinates are normalized between -1 and 1 where
    (0, 0) is the center of the image.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_grid(self, height: int, width: int) -> torch.Tensor:
        """Get the X and Y coordinates for a given height and width.

        Args:
            height (int): Height of the image.
            width (int): Width of the image.

        Returns:
            torch.Tensor: (2, H, W)
        """
        return torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, height),
                torch.linspace(-1, 1, width),
                indexing="ij",
            ),
            dim=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        grid = self.get_grid(height, width).to(x.device)
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        x = torch.cat([x, grid], dim=1)
        return x


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
