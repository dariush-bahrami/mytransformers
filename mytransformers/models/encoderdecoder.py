from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ..buildingblocks.layers import GeneralMultiHeadAttentionLayer


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        number_of_layers: int,
        embedding_dimension: int,
        number_of_heads: int,
        attention_dropout_p: float,
        feedforward_dimension: int,
        feedforward_dropout_p: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GeneralMultiHeadAttentionLayer(
                    queries_embedding_dimension=embedding_dimension,
                    keys_embedding_dimension=embedding_dimension,
                    values_embedding_dimension=embedding_dimension,
                    number_of_heads=number_of_heads,
                    attention_dropout_p=attention_dropout_p,
                    feedforward_dimension=feedforward_dimension,
                    feedforward_dropout_p=feedforward_dropout_p,
                )
                for _ in range(number_of_layers)
            ]
        )

    def forward(
        self,
        embeddings: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            embeddings = layer(
                embeddings,
                embeddings,
                embeddings,
                attention_mask=attention_mask,
            )
        return embeddings


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        number_of_layers: int,
        decoder_embedding_dimension: int,
        encoder_embedding_dimension: int,
        number_of_heads: int,
        attention_dropout_p: float,
        feedforward_dimension: int,
        feedforward_dropout_p: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GeneralMultiHeadAttentionLayer(
                    queries_embedding_dimension=decoder_embedding_dimension,
                    keys_embedding_dimension=encoder_embedding_dimension,
                    values_embedding_dimension=encoder_embedding_dimension,
                    number_of_heads=number_of_heads,
                    attention_dropout_p=attention_dropout_p,
                    feedforward_dimension=feedforward_dimension,
                    feedforward_dropout_p=feedforward_dropout_p,
                )
                for _ in range(number_of_layers)
            ]
        )

    def forward(
        self,
        decoder_embeddings: Tensor,
        encoder_output_embeddings: Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            decoder_embeddings = layer(
                decoder_embeddings,
                encoder_output_embeddings,
                encoder_output_embeddings,
                attention_mask=attention_mask,
            )
        return decoder_embeddings


@dataclass
class EncoderDecoderTransformerConfig:
    number_of_encoder_layers: int
    number_of_decoder_layers: int
    encoder_embedding_dimension: int
    decoder_embedding_dimension: int
    number_of_encoder_heads: int
    number_of_decoder_heads: int
    encoder_feedforward_dimension: int
    decoder_feedforward_dimension: int
    encoder_attention_dropout_p: float
    decoder_attention_dropout_p: float
    encoder_feedforward_dropout_p: float
    decoder_feedforward_dropout_p: float


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, config: EncoderDecoderTransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = TransformerEncoder(
            number_of_layers=config.number_of_encoder_layers,
            embedding_dimension=config.encoder_embedding_dimension,
            number_of_heads=config.number_of_encoder_heads,
            attention_dropout_p=config.encoder_attention_dropout_p,
            feedforward_dimension=config.encoder_feedforward_dimension,
            feedforward_dropout_p=config.encoder_feedforward_dropout_p,
        )
        self.decoder = TransformerDecoder(
            number_of_layers=config.number_of_decoder_layers,
            decoder_embedding_dimension=config.decoder_embedding_dimension,
            encoder_embedding_dimension=config.encoder_embedding_dimension,
            number_of_heads=config.number_of_decoder_heads,
            attention_dropout_p=config.decoder_attention_dropout_p,
            feedforward_dimension=config.decoder_feedforward_dimension,
            feedforward_dropout_p=config.decoder_feedforward_dropout_p,
        )

    def forward(
        self,
        encoder_inputs_embeddings: Tensor,
        decoder_inputs_embeddings: Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tensor:
        encoder_output_embeddings = self.encoder(
            encoder_inputs_embeddings,
            encoder_attention_mask,
        )
        decoder_output_embeddings = self.decoder(
            decoder_inputs_embeddings,
            encoder_output_embeddings,
            decoder_attention_mask,
        )
        return decoder_output_embeddings
