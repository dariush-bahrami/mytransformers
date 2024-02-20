import torch


def get_attention_bias_from_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Get the attention bias from the attention mask. In the attention mask the True
    values are the ones that are not masked and the False values are the ones that are
    masked. The attention bias is a tensor with the same shape as the attention mask
    with -inf in the positions that are masked and 0 in the positions that are not
    masked.

    Args:
        attention_mask (torch.Tensor): (B, S1, S2) or (S1, S2) tensor with the attention
            mask.

    Returns:
        torch.Tensor: (B, S1, S2) or (S1, S2) tensor with the attention bias.
    """
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


def get_attention_masks_from_ignore_mask(ignore_mask: torch.Tensor) -> torch.Tensor:
    """Get the attention mask from the ignore mask. The ignore mask is a tensor with
    the same shape as the attention mask with True values in the positions that are
    masked and False values in the positions that are not masked. The attention mask is
    a tensor with the same shape as the attention mask with True values in the
    positions that are not masked and False values in the positions that are masked.

    Args:
        ignore_mask (torch.Tensor): (B, S) Boolean tensor with the ignore mask. True
            values are the ones that are masked and False values are the ones that are
            not masked.

    Returns:
        torch.Tensor: (B, S, S) Boolean tensor with the attention mask. True values are
            the ones that are not masked and False values are the ones that are masked.
    """
    not_ignore_mask = ignore_mask.logical_not().float()
    attention_mask = torch.bmm(
        not_ignore_mask.unsqueeze(2), not_ignore_mask.unsqueeze(1)
    ).to(dtype=torch.bool, device=ignore_mask.device)
    # make ignore tokens attend to themselves to avoid NaN in softmax
    batch_indices, sequence_indices = ignore_mask.nonzero(as_tuple=True)
    attention_mask[batch_indices, sequence_indices, sequence_indices] = True
    return attention_mask
