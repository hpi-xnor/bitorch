Module bitorch.datasets
=======================

Sub-modules
-----------
* bitorch.datasets.base
* bitorch.datasets.cifar
* bitorch.datasets.dummy_dataset
* bitorch.datasets.imagenet
* bitorch.datasets.mnist

Functions
---------

    
`dataset_from_name(name: str) ‑> Type[bitorch.datasets.base.BasicDataset]`
:   returns the dataset to which the name belongs to (name has to be the value of the datasets
    name-attribute)
    
    Args:
        name (str): name of the dataset
    
    Raises:
        ValueError: raised if no dataset under that name was found
    
    Returns:
        dataset: the dataset

    
`dataset_names() ‑> List[str]`
:   getter for list of dataset names for argparse
    
    Returns:
        List: the dataset names

Classes
-------

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

    `get_dummy_train_and_test_loaders(batch_size: int) ‑> Tuple[bitorch.datasets.dummy_dataset.DummyDataset, bitorch.datasets.dummy_dataset.DummyDataset]`
    :   creates train and test dataloaders for the given dataset. containing example data to test your setup 
        
        Args:
            batch_size (int): batch size of dummy data
        
        Returns:
            Tuple[DummyDataset, DummyDataset]: the generated dataloaders with dummy data. has the same api (but limited) as torch.utils.DataLoader

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

`CIFAR10(train: bool, root_directory: str = None, download: bool = False, augmentation: bitorch.datasets.base.Augmentation = Augmentation.DEFAULT)`
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

    * bitorch.datasets.cifar.CIFAR
    * bitorch.datasets.base.BasicDataset
    * torch.utils.data.dataset.Dataset
    * typing.Generic
    * abc.ABC

    ### Class variables

    `mean: Any`
    :

    `name`
    :

    `num_classes`
    :

    `std_dev: Any`
    :

`CIFAR100(train: bool, root_directory: str = None, download: bool = False, augmentation: bitorch.datasets.base.Augmentation = Augmentation.DEFAULT)`
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

    * bitorch.datasets.cifar.CIFAR
    * bitorch.datasets.base.BasicDataset
    * torch.utils.data.dataset.Dataset
    * typing.Generic
    * abc.ABC

    ### Class variables

    `mean: Any`
    :

    `name`
    :

    `num_classes`
    :

    `std_dev: Any`
    :

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

`MNIST(train: bool, root_directory: str = None, download: bool = False, augmentation: bitorch.datasets.base.Augmentation = Augmentation.DEFAULT)`
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