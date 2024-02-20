import numpy as np
import torch
import torch.nn as nn


def get_natural_sawtooth_wave(times: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """Get the natural sawtooth wave. By natural, we mean that both times and periods
    belong to the Natural numbers (1, 2, 3, ...). The wave at each period will reach its
    maximum 1 step before the next period starts. The wave is between -1 and 1.

    Args:
        times (np.ndarray): (N,) array of times.
        periods (np.ndarray): (M,) array of periods.

    Returns:
        np.ndarray: (N, M) array of the natural sawtooth wave.
    """
    times = times[..., None]
    periods = periods[None, ...]
    slopes = 2 / periods
    result = slopes * (times % periods) - 1
    minimums = np.zeros_like(slopes) - 1
    maximums = slopes * (periods - 1) - 1
    result = (result - minimums) / (maximums - minimums) * 2 - 1
    return result


def get_primes(n: int) -> np.ndarray:
    """Get the first n prime numbers.

    Args:
        n (int): The number of primes to get.

    Returns:
        np.ndarray: (n,) array of the first n prime numbers.
    """
    primes = []
    i = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if i % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
        i += 1
    return np.array(primes)


def get_positional_embeddings(sequence_length: int, embedding_dimension: int):
    """Get positional embeddings for a sequence of a given length and embedding dimension.

    Args:
        sequence_length (int): The length of the sequence.
        embedding_dimension (int): The dimension of the embeddings.

    Returns:
        np.ndarray: (sequence_length, embedding_dimension) array of positional embeddings.
    """
    times = np.arange(1, sequence_length + 1)
    # periods = get_primes(embedding_dimension)
    periods = np.arange(2, embedding_dimension + 2)
    return get_natural_sawtooth_wave(times, periods)


class SawtoothPositionalEncoder(nn.Module):
    def __init__(self, embedding_dimension: int, max_sequence_length: int):
        super().__init__()
        self.register_buffer(
            "positional_encodings",
            torch.from_numpy(
                get_positional_embeddings(max_sequence_length, embedding_dimension)
            ).float(),
        )

    def forward(self, embeddings: torch.Tensor):
        batch_size, sequence_length, embedding_dimension = embeddings.shape
        embeddings = embeddings + self.positional_encodings[:sequence_length]
        return embeddings
