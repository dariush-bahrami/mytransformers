import torch
from torch import nn

from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward
from .utils import get_shifted_right_attention_mask


class EncoderLayer(nn.Module):
    """Encoder layer. Same as the original Transformer encoder layer in the paper
    attention is all you need.

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
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.attenion = MultiHeadAttention(
            embedding_dimension,
            number_of_heads,
            attention_dropout_p,
        )
        self.feedforward = PositionWiseFeedForward(
            embedding_dimension,
            feedforward_dimension,
            feedforward_dropout_p,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the encoder on the embeddings.

        Args:
            embeddings (torch.Tensor): (B, S, E)

        Returns:
            torch.Tensor: (B, S, E)
        """
        query = key = value = self.layer_norm_1(embeddings)
        embeddings = self.attenion(query, key, value) + embeddings
        embeddings = self.feedforward(self.layer_norm_2(embeddings)) + embeddings
        return embeddings


class DecoderLayer(nn.Module):
    """Decoder layer. Same as the original Transformer decoder layer in the paper
    attention is all you need.

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
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.layer_norm_3 = nn.LayerNorm(embedding_dimension)
        self.attenion_1 = MultiHeadAttention(
            embedding_dimension,
            number_of_heads,
            attention_dropout_p,
        )
        self.attenion_2 = MultiHeadAttention(
            embedding_dimension,
            number_of_heads,
            attention_dropout_p,
        )
        self.feedforward = PositionWiseFeedForward(
            embedding_dimension,
            feedforward_dimension,
            feedforward_dropout_p,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        encoder_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the decoder on the embeddings.

        Args:
            embeddings (torch.Tensor): (B, S1, E)
            encoder_outputs (torch.Tensor): (B, S2, E)

        Returns:
            torch.Tensor: (B, S1, E)
        """
        # First attention block (masked self-attention)
        batch_size, sequence_length, embedding_dimension = embeddings.shape
        query = key = value = self.layer_norm_1(embeddings)
        attention_mask = get_shifted_right_attention_mask(
            batch_size, sequence_length
        ).to(embeddings.device)
        embeddings = self.attenion_1(query, key, value, attention_mask) + embeddings
        # Second attention block (encoder-decoder attention)
        query = self.layer_norm_2(embeddings)
        key = value = encoder_outputs
        attention_mask = None
        embeddings = self.attenion_2(query, key, value, attention_mask) + embeddings
        # Feedforward block
        embeddings = self.feedforward(self.layer_norm_3(embeddings)) + embeddings
        return embeddings


class CausalLayer(nn.Module):
    """Causal layer. This module is mostly used in the decoder-only Transformer models
    like GPT-2.

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
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)
        self.attenion = MultiHeadAttention(
            embedding_dimension,
            number_of_heads,
            attention_dropout_p,
        )
        self.feedforward = PositionWiseFeedForward(
            embedding_dimension,
            feedforward_dimension,
            feedforward_dropout_p,
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the causal layer on the embeddings.

        Args:
            embeddings (torch.Tensor): (B, S, E)

        Returns:
            torch.Tensor: (B, S, E)
        """
        batch_size, sequence_length, embedding_dimension = embeddings.shape
        device = embeddings.device
        attention_mask = get_shifted_right_attention_mask(
            batch_size, sequence_length
        ).to(device)
        query = key = value = self.layer_norm_1(embeddings)
        embeddings = self.attenion(query, key, value, attention_mask) + embeddings
        embeddings = self.feedforward(self.layer_norm_2(embeddings)) + embeddings
        return embeddings
