from typing import Optional

import torch
from torch import nn

from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward
from .utils import get_causal_attention_mask


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
            embedding_dimension,
            embedding_dimension,
            embedding_dimension,
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
        encoder_embedding_dimension (int): The embedding dimension of the encoder.
        decoder_embedding_dimension (int): The embedding dimension of the decoder.
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
        encoder_embedding_dimension: int,
        decoder_embedding_dimension: int,
        number_of_heads: int,
        attention_dropout_p: float,
        feedforward_dimension: int,
        feedforward_dropout_p: float,
    ):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(decoder_embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(decoder_embedding_dimension)
        self.layer_norm_3 = nn.LayerNorm(decoder_embedding_dimension)
        self.attenion_1 = MultiHeadAttention(
            decoder_embedding_dimension,
            decoder_embedding_dimension,
            decoder_embedding_dimension,
            decoder_embedding_dimension,
            decoder_embedding_dimension,
            number_of_heads,
            attention_dropout_p,
        )
        self.attenion_2 = MultiHeadAttention(
            decoder_embedding_dimension,
            encoder_embedding_dimension,
            encoder_embedding_dimension,
            decoder_embedding_dimension,
            decoder_embedding_dimension,
            number_of_heads,
            attention_dropout_p,
        )
        self.feedforward = PositionWiseFeedForward(
            decoder_embedding_dimension,
            feedforward_dimension,
            feedforward_dropout_p,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        encoder_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the decoder on the embeddings.

        Args:
            embeddings (torch.Tensor): (B, S1, E1) The E1 is the decoder embedding
                dimension.
            encoder_outputs (torch.Tensor): (B, S2, E2) The E2 is the encoder embedding
                dimension.
            attention_mask (torch.Tensor, optional): (B, S1, S2) The attention mask for
                the first multi-head attention layer. Defaults to None.

        Returns:
            torch.Tensor: (B, S1, E1)
        """
        # First attention block (masked self-attention)
        batch_size, sequence_length, embedding_dimension = embeddings.shape
        query = key = value = self.layer_norm_1(embeddings)
        embeddings = self.attenion_1(query, key, value, attention_mask) + embeddings
        # Second attention block (encoder-decoder attention)
        query = self.layer_norm_2(embeddings)
        key = value = encoder_outputs
        embeddings = self.attenion_2(query, key, value) + embeddings
        # Feedforward block
        embeddings = self.feedforward(self.layer_norm_3(embeddings)) + embeddings
        return embeddings


class GeneralMultiHeadAttentionLayer(nn.Module):
    """General multi-head attention. This module consists of a multi-head attention,
    layer normalization and feedforward layer.

    Args:
        queries_embedding_dimension (int): The embedding dimension of the queries.
        keys_embedding_dimension (int): The embedding dimension of the keys.
        values_embedding_dimension (int): The embedding dimension of the values.
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
        queries_embedding_dimension: int,
        keys_embedding_dimension: int,
        values_embedding_dimension: int,
        number_of_heads: int,
        attention_dropout_p: float,
        feedforward_dimension: int,
        feedforward_dropout_p: float,
    ):
        super().__init__()
        self.queries_layer_norm = nn.LayerNorm(queries_embedding_dimension)
        self.keys_layer_norm = nn.LayerNorm(keys_embedding_dimension)
        self.values_layer_norm = nn.LayerNorm(values_embedding_dimension)
        self.feed_forward_layer_norm = nn.LayerNorm(queries_embedding_dimension)
        self.attention = MultiHeadAttention(
            queries_embedding_dimension,
            keys_embedding_dimension,
            values_embedding_dimension,
            queries_embedding_dimension,
            queries_embedding_dimension,
            number_of_heads,
            attention_dropout_p,
        )
        self.feedforward = PositionWiseFeedForward(
            queries_embedding_dimension,
            feedforward_dimension,
            feedforward_dropout_p,
        )

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the decoder on the embeddings.

        Args:
            queries (torch.Tensor): (B, S1, E1) where E1 is the queries embedding
                dimension.
            keys (torch.Tensor): (B, S2, E2) where E2 is the keys embedding dimension.
            values (torch.Tensor): (B, S2, E3) where E3 is the values embedding
            attention_mask (torch.Tensor, optional): (B, S1, S2) The attention mask for
                the first multi-head attention layer. Defaults to None.

        Returns:
            torch.Tensor: (B, S1, E1)
        """

        embeddings = (
            self.attention(
                self.queries_layer_norm(queries),
                self.keys_layer_norm(keys),
                self.values_layer_norm(values),
                attention_mask,
            )
            + queries
        )
        # Feedforward block
        embeddings = (
            self.feedforward(self.feed_forward_layer_norm(embeddings)) + embeddings
        )
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
            embedding_dimension,
            embedding_dimension,
            embedding_dimension,
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
        attention_mask = get_causal_attention_mask(sequence_length, sequence_length).to(
            device
        )
        query = key = value = self.layer_norm_1(embeddings)
        embeddings = self.attenion(query, key, value, attention_mask) + embeddings
        embeddings = self.feedforward(self.layer_norm_2(embeddings)) + embeddings
        return embeddings
