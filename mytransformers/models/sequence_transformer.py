from typing import Optional

import torch
from torch import Tensor, nn

from ..buildingblocks.layers import GeneralMultiHeadAttentionLayer


class MultiHeadSelfAttentionLayer(GeneralMultiHeadAttentionLayer):
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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the encoder on the embeddings.

        Args:
            embeddings (torch.Tensor): (B, S, E)

        Returns:
            torch.Tensor: (B, S, E)
        """
        super().forward(embeddings, embeddings, embeddings, attention_mask)


class SequenceTransformer(nn.Module):
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
        attention_dropout_p: float,
        feedforward_dropout_p: float,
    ):

        super().__init__()
        self.layers = nn.Sequential(
            *[
                MultiHeadSelfAttentionLayer(
                    embedding_dimension,
                    number_of_heads,
                    attention_dropout_p,
                    feedforward_dimension,
                    feedforward_dropout_p,
                )
                for _ in range(number_of_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x (Tensor): (B, S, E)

        Returns:
            Tensor: (B, S, E)
        """
        return self.layers(x)
