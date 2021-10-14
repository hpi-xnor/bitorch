from typing import Union, Tuple

import torch


class DummyDataset(object):
    """An iterator that produces repeated dummy data.
    Args:
        data_sample: a data sample that should be produced at each step.
        batch_size: the batch size for storing.
        sample_count: number of `data` samples in the dummy dataset.
    """

    def __init__(self, data_sample: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                 batch_size: int, sample_count: int):
        self._data_sample = data_sample
        self._sample_count = sample_count
        self.batch_size = batch_size
        self._count = 0

    def __iter__(self) -> "DummyDataset":
        return DummyDataset(self._data_sample, self.batch_size, self._sample_count)

    def __len__(self) -> int:
        return self._sample_count

    def __next__(self) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        return self.next()

    def next(self) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._count >= self._sample_count:
            raise StopIteration
        self._count += 1
        return self._data_sample
