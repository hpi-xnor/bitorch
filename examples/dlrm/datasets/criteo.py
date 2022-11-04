import gc
from typing import Tuple
from torch.utils.data import Dataset
import logging
import torch
import os
import numpy as np
from .base import BasicDataset
from facebook_dataloading.dataloading_fb import CriteoDataset


class SplitCriteoDataset(Dataset):
    """Dataset to get items from a dataset for each split. Useful if dataset creation takes a lot of time and can be done exactly once."""

    def __init__(
        self, dataset: BasicDataset, split: str, train_split_fraction: float = 0.9, ignore_size: float = 0.0
    ) -> None:
        self.dataset = dataset
        self.indices = self.dataset.train_indices if split == "train" else self.dataset.test_indices

        dataset_size = int(len(self.indices) * (1.0 - ignore_size))
        self.indices = np.random.choice(self.indices, size=dataset_size, replace=False)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dataset[self.indices[idx]]

    def __len__(self) -> int:
        return len(self.indices)


class Criteo(BasicDataset):
    name = "criteo"

    num_train_samples = 60000
    num_val_samples = 10000
    dataset_url = "http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz"

    def get_dataset(self, download: bool = True) -> Dataset:
        try:
            self.download_path = self.root_directory / "criteo.tar.gz"
            self.path = self.root_directory / "train.txt"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if download and not self.download_path.exists():
                logging.info("DOWNLOADING CRITEO DATASET")
                result = os.system(f"wget {self.dataset_url} -O {str(self.root_directory / 'criteo.tar.gz')}")
                if result != 0:
                    raise Exception("Download failed")
                logging.info("FINISHED DOWNLOAD")
            if not (self.root_directory / "train.txt").exists():
                logging.info("EXTRACTING CRITEO DATASET")
                result = os.system(f"tar -xf {str(self.root_directory / 'criteo.tar.gz')} -C {self.root_directory}")
                if result != 0:
                    raise Exception("Extraction failed")
                logging.info("FINISHED EXTRACTION")
        except Exception as e:
            logging.error(
                f"Cannot get criteo dataset: {e}. You need download "
                "the dataset manually under the following link: "
                "http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz "
                f"and extract it to the following path: {str(self.root_directory.resolve())}. "
                "alternatively you can try downloading it automatically by using the --download flag"
            )
        dataset = CriteoDataset(
            dataset="kaggle",
            max_ind_range=-1,
            sub_sample_rate=0.0,
            randomize="total",
            # split="train" if self.is_train else "test",
            raw_path=str(self.root_directory / "train.txt"),
            pro_data=str(self.root_directory / "preprocessed.npz"),
            memory_map=False,
            dataset_multiprocessing=True,
            store_all_indices=True,
        )
        gc.collect()
        return dataset
