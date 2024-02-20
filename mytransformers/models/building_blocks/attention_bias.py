import torch


def get_attention_bias_from_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    attention_bias = torch.zeros_like(attention_mask, dtype=torch.float)
    attention_bias.masked_fill_(attention_mask == False, float("-inf"))
    return attention_bias


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


def get_attention_masks_from_ignore_mask(ignore_mask):
    not_ignore_mask = ignore_mask.logical_not().float()
    attention_mask = torch.bmm(
        not_ignore_mask.unsqueeze(2), not_ignore_mask.unsqueeze(1)
    ).to(dtype=torch.bool, device=ignore_mask.device)
    # make ignore tokens attend to themselves to avoid NaN in softmax
    batch_indices, sequence_indices = ignore_mask.nonzero(as_tuple=True)
    attention_mask[batch_indices, sequence_indices, sequence_indices] = True
    return attention_mask
