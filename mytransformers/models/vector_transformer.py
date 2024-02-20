import math
from typing import Optional, Union

import torch
from torch import nn

from .building_blocks.fully_connected_block import FullyConnectedBlock
from .building_blocks.vector_transformer_block import VectorTransformerBlock


class VectorTransformer(nn.Module):
    """Vector transformer. This module applies a fully connected block to the input
    embeddings, then applies a number of multi head fully connected self attention
    modules in sequence and finally applies a linear transformation to the outputs
    of the multi head fully connected self attention modules.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        embedding_dim: int,
        heads_dim: int,
        num_heads: int,
        num_layers: int,
        fully_connected_dropout_p: float,
        attention_dropout_p: float,
    ) -> None:
        super().__init__()
        # Non learnable input normalizer just to make sure that the input embeddings
        # are normalized.
        self.input_normalizer = nn.BatchNorm1d(in_features, affine=False)
        self.input_layer = FullyConnectedBlock(
            in_features, embedding_dim, fully_connected_dropout_p
        )
        self.hidden_layers = nn.Sequential(
            *[
                VectorTransformerBlock(
                    embedding_dim,
                    heads_dim,
                    num_heads,
                    fully_connected_dropout_p,
                    attention_dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_layer = nn.Linear(embedding_dim, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Vector transformer forward pass.

        Args:
            x (Tensor): (B, E)

        Returns:
            Tensor: (B, E)
        """
        x = self.input_normalizer(x)
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
