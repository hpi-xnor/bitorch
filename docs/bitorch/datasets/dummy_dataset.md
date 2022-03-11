Module bitorch.datasets.dummy_dataset
=====================================

Classes
-------

`DummyDataset(data_sample: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], batch_size: int, sample_count: int)`
:   An iterator that produces repeated dummy data.
    Args:
        data_sample: a data sample that should be produced at each step.
        batch_size: the batch size for storing.
        sample_count: number of `data` samples in the dummy dataset.

    ### Methods

    `next(self) ‑> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]`
    :