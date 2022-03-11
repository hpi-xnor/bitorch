Module bitorch.datasets.dummy_dataset
=====================================

Classes
-------

`DummyDataset(data_shape: torch.Size, num_classes: int, sample_count: int)`
:   An iterator that produces repeated dummy data.
    Args:
        data_sample: a data sample that should be produced at each step.
        batch_size: the batch size for storing.
        sample_count: number of `data` samples in the dummy dataset.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic

    ### Class variables

    `functions: Dict[str, Callable]`
    :