import torch
from torch import nn


def get_causal_attention_mask(
    query_sequence_length: int,
    key_value_sequence_length: int,
) -> torch.Tensor:
    """Get the shifted right attention mask. This is used to mask out the future tokens
    in the sequence.

    Args:
        query_sequence_length (int): Query sequence length (S1).
        key_value_sequence_length (int): Key and value sequence length (S2).

    Returns:
        torch.Tensor: (B, S1, S2) tensor with the causal mask.
    """
    mask = torch.ones(
        (query_sequence_length, key_value_sequence_length), dtype=torch.bool
    )
    mask = torch.tril(mask)
    return mask
