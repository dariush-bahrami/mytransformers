import torch
from torch import nn


def get_shifted_right_attention_mask(
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    """Get the shifted right attention mask. This is used to mask out the future tokens
    in the sequence.

    Args:
        batch_size (int): Batch size (B).
        sequence_length (int): Sequence length (S).

    Returns:
        torch.Tensor: (B, S, S)
    """
    mask = torch.ones((batch_size, sequence_length, sequence_length), dtype=torch.bool)
    mask = torch.tril(mask)
    return mask
