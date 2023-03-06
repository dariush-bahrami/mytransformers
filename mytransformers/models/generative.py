from dataclasses import dataclass

import torch
from torch import nn

from ..buildingblocks import CausalLayer, LearnablePositionalEncoder, TokenEmbedder


@dataclass
class GenerativeTransformerConfig:
    vocabulary_size: int
    context_size: int
    embedding_dimension: int
    number_of_heads: int
    attention_dropout_p: float
    feedforward_dimension: int
    feedforward_dropout_p: float
    number_of_layers: int
    positional_encoder_dropout_p: float


class GenerativeTransformer(nn.Module):
    def __init__(self, config: GenerativeTransformerConfig):
        super().__init__()
        self.token_embedder = TokenEmbedder(
            config.vocabulary_size, config.embedding_dimension
        )
        self.positional_encoder = LearnablePositionalEncoder(
            config.embedding_dimension,
            config.context_size,
            config.positional_encoder_dropout_p,
        )
        causal_layer_args = (
            config.embedding_dimension,
            config.number_of_heads,
            config.attention_dropout_p,
            config.feedforward_dimension,
            config.feedforward_dropout_p,
        )
        self.causal_layers = nn.Sequential(
            *(CausalLayer(*causal_layer_args) for _ in range(config.number_of_layers))
        )
        self.layer_norm = nn.LayerNorm(config.embedding_dimension)
        self.linear = nn.Linear(config.embedding_dimension, config.vocabulary_size)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embedder(indices)
        embeddings = self.positional_encoder(embeddings)
        embeddings = self.causal_layers(embeddings)
        embeddings = self.layer_norm(embeddings)
        logits = self.linear(embeddings)
        return logits


def get_gpt_model(
    vocabulary_size: int, context_size: int, scale: float
) -> GenerativeTransformer:
    number_of_layers = round(12 * scale)
    embedding_dimension = round(768 * scale)
    number_of_heads = round(12 * scale)
    feedforward_dimension = round(3072 * scale)
    attention_dropout_p = 0.0
    feedforward_dropout_p = 0.0
    positional_encoder_dropout_p = 0.0
    config = GenerativeTransformerConfig(
        vocabulary_size,
        context_size,
        embedding_dimension,
        number_of_heads,
        attention_dropout_p,
        feedforward_dimension,
        feedforward_dropout_p,
        number_of_layers,
        positional_encoder_dropout_p,
    )
    return GenerativeTransformer(config)
