import torch
from torch import nn

from .dot_product_attention import DotProductAttention
from .fully_connected_block import FullyConnectedBlock
from .learnable_sequence_pooling import LearnableSequencePooling


class VectorTransformerBlock(nn.Module):
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
        self.attention = DotProductAttention(dropout_p=attention_dropout_p)
        self.heads_pooling = LearnableSequencePooling(num_heads)
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
