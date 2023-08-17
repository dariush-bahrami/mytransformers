import torch
from torch import nn

from ...buildingblocks.attention import ScaledDotProductAttention
from ...buildingblocks.pooling import FixedLengthWeightedSequencePooling


class FullyConnectedBlock(nn.Module):
    """Fully connected block. This block applies a linear transformation followed
    by a layer normalization, a GELU activation and a dropout.
    """

    def __init__(self, in_features: int, out_features: int, dropout_p: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.normalizer = nn.LayerNorm(out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Fully connected block forward pass.

        Args:
            x (Tensor): (B, E)

        Returns:
            Tensor: (B, E)
        """
        x = self.linear(x)
        x = self.normalizer(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class MultiHeadFullyConnectedSelfAttention(nn.Module):
    """Multi head fully connected self attention. This module applies fully connected
    blocks to the input embeddings in parallel and then applies scaled dot product
    self attention to the outputs of the fully connected blocks. The outputs of the
    scaled dot product self attention are then projected back to the input embedding
    and after that a weighted sum of the outputs of the scaled dot product self
    attention is computed. The weights of the weighted sum are learned.
    """

    def __init__(
        self,
        embedding_dim: int,
        heads_dim: int,
        num_heads: int,
        fully_connected_dropout_p: float,
        attention_dropout_p: float,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [
                FullyConnectedBlock(embedding_dim, heads_dim, fully_connected_dropout_p)
                for _ in range(num_heads)
            ]
        )
        self.attention = ScaledDotProductAttention(dropout_p=attention_dropout_p)
        self.heads_pooling = FixedLengthWeightedSequencePooling(num_heads)
        self.output_projection = nn.Linear(
            heads_dim,
            embedding_dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Multi head fully connected self attention forward pass.

        Args:
            x (Tensor): (B, E)

        Returns:
            Tensor: (B, E)
        """
        # Note: B = batch size, E = input embedding dim, S = sequence length,
        # Nh = number of heads, Eh = head embedding dim
        embeddings = torch.stack([h(x) for h in self.heads], dim=1)  # (B, Nh, Eh)
        embeddings = self.attention(embeddings, embeddings, embeddings)  # (B, Nh, Eh)
        embeddings = self.output_projection(embeddings)  # (B, Nh, E)
        embeddings = self.heads_pooling(embeddings)  # (B, 1, E)
        embeddings = embeddings.squeeze(dim=1)  # (B, E)
        return x + embeddings


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
                MultiHeadFullyConnectedSelfAttention(
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
