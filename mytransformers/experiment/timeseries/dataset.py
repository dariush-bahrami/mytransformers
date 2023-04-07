from typing import TypedDict

from torch import Tensor
from torch.utils.data import Dataset


class DataDict(TypedDict):
    past_time: Tensor
    past_context: Tensor
    past_target: Tensor
    future_time: Tensor
    future_target: Tensor


class TimeSeriesTransformerDataset(Dataset):
    def __init__(
        self,
        time: Tensor,
        context: Tensor,
        target: Tensor,
        past_sequence_length: int,
        future_sequence_length: int,
    ) -> None:
        self.time = time
        self.context = context
        self.target = target
        self.past_sequence_length = past_sequence_length
        self.future_sequence_length = future_sequence_length
        self.__length = len(time) - (past_sequence_length + future_sequence_length) + 1

    def __len__(self) -> int:
        return self.__length

    def __getitem__(self, idx: int) -> DataDict:
        # Calculate the indices for the past and future sequences
        past_idx_start = idx
        past_idx_end = idx + self.past_sequence_length
        future_idx_start = past_idx_end
        future_idx_end = past_idx_end + self.future_sequence_length

        return {
            "past_time": self.time[past_idx_start:past_idx_end],
            "past_context": self.context[past_idx_start:past_idx_end],
            "past_target": self.target[past_idx_start:past_idx_end],
            "future_time": self.time[future_idx_start:future_idx_end],
            "future_target": self.target[future_idx_start:future_idx_end],
        }
