from typing import Optional

import torch
from torch import nn

from .attention_bias import (
    get_attention_bias_from_attention_mask,
    get_attention_masks_from_ignore_mask,
    get_causal_attention_mask,
)
from .sequence_transformer_block import SequenceTransformerBlock


class BidirectionalSequenceTransformerBlock(SequenceTransformerBlock):
    """Multi-head self-attention layer. This module consists of a multi-head attention,
    layer normalization and feedforward layer. The queries, keys and values are all the
    same.

    Args:
        embedding_dimension (int): The embedding dimension of the input.
        number_of_heads (int): The number of attention heads.
        attention_dropout_p (float): The probability of dropping out a value in the
            attention weights.
        feedforward_dimension (int): The dimension of the intermediate layer. A simple
            rule of thumb is to set this to 4 times the embedding dimension.
        feedforward_dropout_p (float): The probability of dropping out a value in the
            feedforward layer.
    """

    def __init__(
        self,
        embedding_dimension: int,
        number_of_heads: int,
        attention_dropout_p: float,
        feedforward_dimension: int,
        feedforward_dropout_p: float,
    ):
        super().__init__(
            embedding_dimension,
            embedding_dimension,
            embedding_dimension,
            number_of_heads,
            attention_dropout_p,
            feedforward_dimension,
            feedforward_dropout_p,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        ignore_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the encoder on the embeddings.

        Args:
            embeddings (torch.Tensor): (B, S, E)

        Returns:
            torch.Tensor: (B, S, E)
        """
        if ignore_mask is not None:
            if ignore_mask.any():
                attention_mask = get_attention_masks_from_ignore_mask(ignore_mask)
                attention_bias = get_attention_bias_from_attention_mask(
                    attention_mask
                ).to(
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                )
        super().forward(
            embeddings, embeddings, embeddings, attention_bias=attention_bias
        )
