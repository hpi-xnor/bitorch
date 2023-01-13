# fmt: off
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the dlrm benchmark
# The inpts and outputs are generated according to the following three option(s)
# 1) random distribution
# 2) synthetic distribution, based on unique accesses and distances between them
#    i) R. Hassan, A. Harris, N. Topham and A. Efthymiou "Synthetic Trace-Driven
#    Simulation of Cache Memory", IEEE AINAM'07
# 3) public data set
#    i)  Criteo Kaggle Display Advertising Challenge Dataset
#    https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
#    ii) Criteo Terabyte Dataset
#    https://labs.criteo.com/2013/12/download-terabyte-click-logs


from __future__ import absolute_import, division, print_function, unicode_literals
import gc
# others
from os import path
import sys
import logging
from typing import Any, List, Tuple

from . import data_utils

# numpy
import numpy as np

# pytorch
import torch
from torch.utils.data import Dataset


# Kaggle Display Advertising Challenge Dataset
# dataset (str): name of dataset (Kaggle or Terabyte)
# randomize (str): determines randomization scheme
#            "none": no randomization
#            "day": randomizes each day"s data (only works if split = True)
#            "total": randomizes total dataset
# split (bool) : to split into train, test, validation data-sets
class CriteoDataset(Dataset):

    def __init__(
            self,
            dataset: str,
            max_ind_range: int,
            sub_sample_rate: int,
            randomize: bool,
            split: str = "train",
            raw_path: str = "",
            pro_data: str = "",
            memory_map: str = False,
            dataset_multiprocessing: str = False,
            store_all_indices: str = False,
    ) -> None:
        # dataset
        # tar_fea = 1   # single target
        den_fea = 13  # 13 dense  features
        # spa_fea = 26  # 26 sparse features
        # tad_fea = tar_fea + den_fea
        # tot_fea = tad_fea + spa_fea
        if dataset == "kaggle":
            days = 7
            out_file = "kaggleAdDisplayChallenge_processed"
        elif dataset == "terabyte":
            days = 24
            out_file = "terabyte_processed"
        else:
            raise (ValueError("Data set option is not supported"))
        self.max_ind_range = max_ind_range
        self.memory_map = memory_map

        # split the datafile into path and filename
        lstr = raw_path.split("/")
        self.d_path = "/".join(lstr[0:-1]) + "/"
        self.d_file = lstr[-1].split(".")[0] if dataset == "kaggle" else lstr[-1]
        self.npzfile = self.d_path + (
            (self.d_file + "_day") if dataset == "kaggle" else self.d_file
        )
        self.trafile = self.d_path + (
            (self.d_file + "_fea") if dataset == "kaggle" else "fea"
        )

        # check if pre-processed data is available
        data_ready = True
        if memory_map:
            for i in range(days):
                reo_data = self.npzfile + "_{0}_reordered.npz".format(i)
                if not path.exists(str(reo_data)):
                    data_ready = False
        else:
            if not path.exists(str(pro_data)):
                data_ready = False

        # pre-process data if needed
        # WARNNING: when memory mapping is used we get a collection of files
        if data_ready:
            logging.debug("Reading pre-processed data=%s" % (str(pro_data)))
            file = str(pro_data)
        else:
            logging.debug("Reading raw data=%s" % (str(raw_path)))
            file = data_utils.getCriteoAdData(
                raw_path,
                out_file,
                max_ind_range,
                sub_sample_rate,
                days,
                split,
                randomize,
                dataset == "kaggle",
                memory_map,
                dataset_multiprocessing,
            )

        # get a number of samples per day
        total_file = self.d_path + self.d_file + "_day_count.npz"
        with np.load(total_file) as data:
            total_per_file = data["total_per_file"]
        # compute offsets per file
        self.offset_per_file = np.array([0] + [x for x in total_per_file])
        for i in range(days):
            self.offset_per_file[i + 1] += self.offset_per_file[i]
        # logging.debug(self.offset_per_file)

        # setup data
        if memory_map:
            # setup the training/testing split
            self.split = split
            if split == 'none' or split == 'train':
                self.day = 0
                self.max_day_range = days if split == 'none' else days - 1
            elif split == 'test' or split == 'val':
                self.day = days - 1
                num_samples = self.offset_per_file[days] - self.offset_per_file[days - 1]
                self.test_size = int(np.ceil(num_samples / 2.))
                self.val_size = num_samples - self.test_size
            else:
                sys.exit("ERROR: dataset split is neither none, nor train or test.")

            # load unique counts
            with np.load(self.d_path + self.d_file + "_fea_count.npz") as data:
                self.counts = data["counts"]
            self.m_den = den_fea  # X_int.shape[1]
            self.n_emb = len(self.counts)
            logging.debug("Sparse features= %d, Dense features= %d" % (self.n_emb, self.m_den))

            # Load the test data
            # Only a single day is used for testing
            if self.split == 'test' or self.split == 'val':
                # only a single day is used for testing
                fi = self.npzfile + "_{0}_reordered.npz".format(
                    self.day
                )
                with np.load(fi) as data:
                    self.X_int = data["X_int"]  # continuous  feature
                    self.X_cat = data["X_cat"]  # categorical feature
                    self.y = data["y"]          # target

        else:
            # load and preprocess data
            with np.load(file) as data:
                X_int = data["X_int"]  # continuous  feature
                X_cat = data["X_cat"]  # categorical feature
                y = data["y"]          # target
                self.counts = data["counts"]
            self.m_den = X_int.shape[1]  # den_fea
            self.n_emb = len(self.counts)
            logging.debug("Sparse fea = %d, Dense fea = %d" % (self.n_emb, self.m_den))

            # create reordering
            indices = np.arange(len(y))

            self.train_indices = None
            self.test_indices = None
            self.val_indices = None
            if store_all_indices:
                indices = np.array_split(indices, self.offset_per_file[1:-1])

                # randomize train data (per day)
                if randomize == "day":  # or randomize == "total":
                    for i in range(len(indices) - 1):
                        indices[i] = np.random.permutation(indices[i])
                    logging.debug("Randomized indices per day ...")

                self.train_indices = np.concatenate(indices[:-1])
                self.test_indices = indices[-1]
                self.test_indices, self.val_indices = np.array_split(self.test_indices, 2)

                logging.debug("Defined %s indices..." % (split))
                self.X_int = X_int
                self.X_cat = X_cat
                self.y = y
                # randomize train data (across days)
                if randomize == "total":
                    train_indices = np.random.permutation(self.train_indices)
                    logging.debug("Randomized indices across days ...")

            elif split == "none":
                # randomize all data
                if randomize == "total":
                    indices = np.random.permutation(indices)
                    logging.debug("Randomized indices...")

                X_int[indices] = X_int
                X_cat[indices] = X_cat
                y[indices] = y

            else:
                indices = np.array_split(indices, self.offset_per_file[1:-1])

                # randomize train data (per day)
                if randomize == "day":  # or randomize == "total":
                    for i in range(len(indices) - 1):
                        indices[i] = np.random.permutation(indices[i])
                    logging.debug("Randomized indices per day ...")

                train_indices = np.concatenate(indices[:-1])
                test_indices = indices[-1]
                test_indices, val_indices = np.array_split(test_indices, 2)

                logging.debug("Defined %s indices..." % (split))

                # randomize train data (across days)
                if randomize == "total":
                    train_indices = np.random.permutation(train_indices)
                    logging.debug("Randomized indices across days ...")

                # create training, validation, and test sets
                if split == 'train':
                    self.X_int = [X_int[i] for i in train_indices]
                    self.X_cat = [X_cat[i] for i in train_indices]
                    self.y = [y[i] for i in train_indices]
                elif split == 'val':
                    self.X_int = [X_int[i] for i in val_indices]
                    self.X_cat = [X_cat[i] for i in val_indices]
                    self.y = [y[i] for i in val_indices]
                elif split == 'test':
                    self.X_int = [X_int[i] for i in test_indices]
                    self.X_cat = [X_cat[i] for i in test_indices]
                    self.y = [y[i] for i in test_indices]

            logging.debug("Split data according to indices...")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        if self.memory_map:
            if self.split == 'none' or self.split == 'train':
                # check if need to swicth to next day and load data
                if index == self.offset_per_file[self.day]:
                    # logging.debug("day_boundary switch", index)
                    self.day_boundary = self.offset_per_file[self.day]
                    fi = self.npzfile + "_{0}_reordered.npz".format(
                        self.day
                    )
                    # logging.debug('Loading file: ', fi)
                    with np.load(fi) as data:
                        self.X_int = data["X_int"]  # continuous  feature
                        self.X_cat = data["X_cat"]  # categorical feature
                        self.y = data["y"]          # target
                    self.day = (self.day + 1) % self.max_day_range

                i = index - self.day_boundary
            elif self.split == 'test' or self.split == 'val':
                # only a single day is used for testing
                i = index + (0 if self.split == 'test' else self.test_size)
            else:
                sys.exit("ERROR: dataset split is neither none, nor train or test.")
        else:
            i = index

        if self.max_ind_range > 0:
            return self.X_int[i], self.X_cat[i] % self.max_ind_range, self.y[i]
        else:
            return self.X_int[i], self.X_cat[i], self.y[i]

    def _default_preprocess(
            self,
            X_int: torch.Tensor,
            X_cat: torch.Tensor,
            y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X_int = torch.log(torch.tensor(X_int, dtype=torch.float) + 1)
        if self.max_ind_range > 0:
            X_cat = torch.tensor(X_cat % self.max_ind_range, dtype=torch.long)
        else:
            X_cat = torch.tensor(X_cat, dtype=torch.long)
        y = torch.tensor(y.astype(np.float32))

        return X_int, X_cat, y

    def __len__(self) -> int:
        if self.memory_map:
            if self.split == 'none':
                return self.offset_per_file[-1]
            elif self.split == 'train':
                return self.offset_per_file[-2]
            elif self.split == 'test':
                return self.test_size
            elif self.split == 'val':
                return self.val_size
            else:
                sys.exit("ERROR: dataset split is neither none, nor train nor test.")
        else:
            return len(self.y)


def collate_wrapper_criteo_offset(
        list_of_tuples: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """collates the input features into processabel tensors

    Args:
        list_of_tuples (List[torch.Tensor]): input tensors

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: output
    """
    # where each tuple is (X_int, X_cat, y)
    # transposed_data = np.array(list(zip(*list_of_tuples)), dtype=np.float32)
    # transposed_data = list(zip(*list_of_tuples))

    transposed_data = list(np.array(i) for i in zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = [X_cat[:, i] for i in range(featureCnt)]
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]
    gc.collect()
    return X_int, torch.stack(lS_i), torch.stack(lS_o), T


# Conversion from offset to length
def offset_to_length_converter(lS_o: torch.Tensor, lS_i: torch.Tensor) -> torch.Tensor:
    """converts the offsets of categorical features into tensors containing length

    Args:
        lS_o (torch.Tensor): offset tensors
        lS_i (torch.Tensor): indices

    Returns:
        torch.Tensor: lengths
    """
    def diff(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[1:] - tensor[:-1]

    return torch.stack(
        [
            diff(torch.cat((S_o, torch.tensor(lS_i[ind].shape))).int())
            for ind, S_o in enumerate(lS_o)
        ]
    )


def collate_wrapper_criteo_length(
        list_of_tuples: List[Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """collates the input features into processabel tensors

    Args:
        list_of_tuples (List[torch.Tensor]): input tensors

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: output
    """
    transposed_data = list(np.array(i) for i in zip(*list_of_tuples))
    # transposed_data = np.array(list(zip(*list_of_tuples)), dtype=np.float32)
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = torch.stack([X_cat[:, i] for i in range(featureCnt)])
    lS_o = torch.stack(
        [torch.tensor(range(batchSize)) for _ in range(featureCnt)]
    )

    lS_l = offset_to_length_converter(lS_o, lS_i)

    return X_int, lS_l, lS_i, T
