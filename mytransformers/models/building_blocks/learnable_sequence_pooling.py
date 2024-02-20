import math

import torch
from torch import nn


class LearnableSequencePooling(nn.Module):
    def __init__(self, input_sequence_length: int):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(1, input_sequence_length))
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Weighted sequence pooling forward pass.

        Args:
            x (Tensor): (B, S, E)

        Returns:
            Tensor: (B, E)
        """
        return self.weights @ x
