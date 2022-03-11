Module bitorch.datasets.imagenet
================================

Classes
-------

`ImageNet(train: bool, root_directory: str = None, download: bool = False, augmentation: bitorch.datasets.base.Augmentation = Augmentation.DEFAULT)`
:   An abstract class representing a :class:`Dataset`.
    
    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.
    
    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    
    initializes the dataset.
    
    Args:
        train (bool): whether the train or test dataset is wanted
        root_directory (str): path to main dataset storage directory
        download (bool): whether train/test should be downloaded if it does not exist
        augmentation (Augmentation): the level of augmentation (only for train dataset)
    
    Returns:
        Dataset: the created test/train dataset

    ### Ancestors (in MRO)

    * bitorch.datasets.base.BasicDataset
    * torch.utils.data.dataset.Dataset
    * typing.Generic

    ### Class variables

    `mean: Any`
    :

    `name`
    :

    `num_classes`
    :

    `num_train_samples`
    :

    `num_val_samples`
    :

    `shape`
    :

    `std_dev: Any`
    :

    ### Methods

    `get_data_dir(self) ‑> pathlib.Path`
    :