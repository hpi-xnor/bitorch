Module bitorch.datasets.base
============================

Classes
-------

`Augmentation(value, names=None, *, module=None, qualname=None, type=None, start=1)`
:   An enumeration.

    ### Ancestors (in MRO)

    * enum.Enum

    ### Class variables

    `DEFAULT`
    :

    `HIGH`
    :

    `LOW`
    :

    `MEDIUM`
    :

    `NONE`
    :

    ### Static methods

    `from_string(level: str) ‑> bitorch.datasets.base.Augmentation`
    :

`BasicDataset(train: bool, root_directory: str = None, download: bool = False, augmentation: bitorch.datasets.base.Augmentation = Augmentation.DEFAULT)`
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

    * torch.utils.data.dataset.Dataset
    * typing.Generic

    ### Descendants

    * bitorch.datasets.cifar.CIFAR
    * bitorch.datasets.imagenet.ImageNet
    * bitorch.datasets.mnist.MNIST

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

    ### Static methods

    `get_dummy_train_and_test_datasets() ‑> Tuple[bitorch.datasets.dummy_dataset.DummyDataset, bitorch.datasets.dummy_dataset.DummyDataset]`
    :

    `get_normalize_transform() ‑> torchvision.transforms.transforms.Normalize`
    :

    `get_train_and_test(root_directory: str, download: bool = False, augmentation: bitorch.datasets.base.Augmentation = Augmentation.DEFAULT) ‑> Tuple[bitorch.datasets.base.BasicDataset, bitorch.datasets.base.BasicDataset]`
    :   creates a pair of train and test dataset.
        
        Returns:
            Tuple: the train and test dataset

    `test_transform() ‑> Any`
    :   get the transform for the test data.
        
        Returns:
            transform: the transform pipeline

    `train_transform(augmentation: bitorch.datasets.base.Augmentation = Augmentation.DEFAULT) ‑> Any`
    :   get the transform for the training data (should consider the current augmentation_level).
        
        Returns:
            transform: the transform pipeline

    ### Methods

    `get_dataset(self, download: bool) ‑> torch.utils.data.dataset.Dataset`
    :   creates the actual dataset
        
        Args:
            download (bool): toggles if train/test shall be downloaded if possible
        
        Raises:
            NotImplementedError: thrown, because this method needs to be overwritten by subclasses
        
        Returns:
            Dataset: the created test/train dataset

    `get_dataset_root_directory(self, root_directory_argument: Optional[str]) ‑> pathlib.Path`
    :   chooses the dataset root directory based on the passed argument or environment variables.
        
        Returns:
            Tuple: the train and test dataset

    `get_transform(self) ‑> Any`
    :

    `num_samples(self) ‑> int`
    :   returns the (theoretical) dataset size.