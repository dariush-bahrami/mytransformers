from typing import Optional

import torch
from torch import nn

from .building_blocks.bidirectional_sequence_transformer_block import (
    BidirectionalSequenceTransformerBlock,
)
from .building_blocks.sawtooth_positional_encoder import SawtoothPositionalEncoder


class BidirectionalSequenceTransformer(nn.Module):
    """Sequence Transformer. This module consists of a stack of multi-head
    self-attention layers.

    Args:
        number_of_layers (int): The number of layers in the transformer.
        number_of_heads (int): The number of attention heads.
        embedding_dimension (int): The embedding dimension of the input.
        feedforward_dimension (int): The dimension of the intermediate layer. A simple
            rule of thumb is to set this to 4 times the embedding dimension.
        attention_dropout_p (float): The probability of dropping out a value in the
            attention weights.
        feedforward_dropout_p (float): The probability of dropping out a value in the
            feedforward layer.
    """

    def __init__(
        self,
        number_of_layers: int,
        number_of_heads: int,
        embedding_dimension: int,
        feedforward_dimension: int,
        max_sequence_length: int,
        attention_dropout_p: float,
        feedforward_dropout_p: float,
    ):

        super().__init__()
        self.positional_encoder = SawtoothPositionalEncoder(
            embedding_dimension, max_sequence_length
        )
        self.layers = nn.Sequential(
            *[
                BidirectionalSequenceTransformerBlock(
                    embedding_dimension,
                    number_of_heads,
                    attention_dropout_p,
                    feedforward_dimension,
                    feedforward_dropout_p,
                )
                for _ in range(number_of_layers)
            ]
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        ignore_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            embeddings (Tensor): (B, S, E)
            ignore_mask (torch.Tensor, optional): (B, S) tensor with the ignore mask.
                This mask is used to mask out the padding tokens or any other tokens
                that should be ignored. Defaults to None. If None, no mask is applied.
        Returns:
            Tensor: (B, S, E)
        """
        embeddings = self.positional_encoder(embeddings)
        for layer in self.layers:
            embeddings = layer(embeddings, ignore_mask)
        return embeddings
