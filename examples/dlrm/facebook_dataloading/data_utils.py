# fmt: off
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the DLRM benchmark
#
# Utility function(s) to download and pre-process public data sets
#   - Criteo Kaggle Display Advertising Challenge Dataset
#     https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
#   - Criteo Terabyte Dataset
#     https://labs.criteo.com/2013/12/download-terabyte-click-logs
#
# After downloading dataset, run:
#   getCriteoAdData(
#       datafile="<path-to-train.txt>",
#       o_filename=kaggleAdDisplayChallenge_processed.npz,
#       max_ind_range=-1,
#       sub_sample_rate=0.0,
#       days=7,
#       data_split='train',
#       randomize='total',
#       criteo_kaggle=True,
#       memory_map=False
#   )
#   getCriteoAdData(
#       datafile="<path-to-day_{0,...,23}>",
#       o_filename=terabyte_processed.npz,
#       max_ind_range=-1,
#       sub_sample_rate=0.0,
#       days=24,
#       data_split='train',
#       randomize='total',
#       criteo_kaggle=False,
#       memory_map=False
#   )

from __future__ import absolute_import, division, print_function, unicode_literals

from pathlib import Path
import sys
# import os
import logging
from os import path
from multiprocessing import Process, Manager
from typing import Optional, Any, Union
# import io
# from io import StringIO
# import collections as coll

import numpy as np


def processCriteoAdData(
        d_path: str,
        d_file: str,
        npzfile: str,
        i: int,
        convertDicts: dict,
        pre_comp_counts: bool) -> None:
    """Process Kaggle Display Advertising Challenge or Terabyte Dataset
    by converting unicode strings in X_cat to integers and
    converting negative integer values in X_int.

    Loads data in the form "{kaggle|terabyte}_day_i.npz" where i is the day.

    Args:
        d_path (str): path for {kaggle|terabyte}_day_i.npz files
        d_file (str): _description_
        npzfile (str): _description_
        i (int): splits in the dataset (typically 0 to 7 or 0 to 24)
        convertDicts (dict): _description_
        pre_comp_counts (bool): _description_
    """
    # process data if not all files exist
    filename_i = npzfile + "_{0}_processed.npz".format(i)

    if path.exists(filename_i):
        logging.debug("Using existing " + filename_i, end="\n")
    else:
        logging.debug("Not existing " + filename_i)
        with np.load(npzfile + "_{0}.npz".format(i)) as data:
            # categorical features
            # Approach 2a: using pre-computed dictionaries
            X_cat_t = np.zeros(data["X_cat_t"].shape)
            for j in range(26):
                for k, x in enumerate(data["X_cat_t"][j, :]):
                    X_cat_t[j, k] = convertDicts[j][x]
            # continuous features
            X_int = data["X_int"]
            X_int[X_int < 0] = 0
            # targets
            y = data["y"]

        np.savez_compressed(
            filename_i,
            # X_cat = X_cat,
            X_cat=np.transpose(X_cat_t),  # transpose of the data
            X_int=X_int,
            y=y,
        )
        logging.debug("Processed " + filename_i, end="\n")

    return


def concatCriteoAdData(
        d_path: Any,
        d_file: Any,
        npzfile: Any,
        trafile: Any,
        days: Any,
        data_split: Any,
        randomize: Any,
        total_per_file: Any,
        total_count: Any,
        memory_map: Any,
        o_filename: Any
) -> str:
    """Concatenates different days and saves the result.

    Args:
      days (int): total number of days in the dataset (typically 7 or 24)
      d_path (str): path for {kaggle|terabyte}_day_i.npz files
      o_filename (str): output file name

    Return:
      o_file (str): output file path
    """
    if memory_map:
        # dataset break up per fea
        # tar_fea = 1   # single target
        den_fea = 13  # 13 dense  features
        spa_fea = 26  # 26 sparse features
        # tad_fea = tar_fea + den_fea
        # tot_fea = tad_fea + spa_fea
        # create offset per file
        offset_per_file = np.array([0] + [x for x in total_per_file])
        for i in range(days):
            offset_per_file[i + 1] += offset_per_file[i]

        # Approach 4: Fisher-Yates-Rao (FYR) shuffle algorithm
        # 1st pass of FYR shuffle
        # check if data already exists
        recreate_flag = False
        for j in range(days):
            filename_j_y = npzfile + "_{0}_intermediate_y.npy".format(j)
            filename_j_d = npzfile + "_{0}_intermediate_d.npy".format(j)
            filename_j_s = npzfile + "_{0}_intermediate_s.npy".format(j)
            if (
                path.exists(filename_j_y)
                and path.exists(filename_j_d)
                and path.exists(filename_j_s)
            ):
                logging.debug(
                    "Using existing\n"
                    + filename_j_y + "\n"
                    + filename_j_d + "\n"
                    + filename_j_s
                )
            else:
                recreate_flag = True
        # reorder across buckets using sampling
        if recreate_flag:
            # init intermediate files (.npy appended automatically)
            for j in range(days):
                filename_j_y = npzfile + "_{0}_intermediate_y".format(j)
                filename_j_d = npzfile + "_{0}_intermediate_d".format(j)
                filename_j_s = npzfile + "_{0}_intermediate_s".format(j)
                np.save(filename_j_y, np.zeros((total_per_file[j])))
                np.save(filename_j_d, np.zeros((total_per_file[j], den_fea)))
                np.save(filename_j_s, np.zeros((total_per_file[j], spa_fea)))
            # start processing files
            total_counter = [0] * days
            for i in range(days):
                filename_i = npzfile + "_{0}_processed.npz".format(i)
                with np.load(filename_i) as data:
                    X_cat = data["X_cat"]
                    X_int = data["X_int"]
                    y = data["y"]
                size = len(y)
                # sanity check
                if total_per_file[i] != size:
                    sys.exit("ERROR: sanity check on number of samples failed")
                # debug prints
                logging.debug("Reordering (1st pass) " + filename_i)

                # create buckets using sampling of random ints
                # from (discrete) uniform distribution
                buckets = []
                for _j in range(days):
                    buckets.append([])
                counter = [0] * days
                days_to_sample = days if data_split == "none" else days - 1
                if randomize == "total":
                    rand_u = np.random.randint(low=0, high=days_to_sample, size=size)
                    for k in range(size):
                        # sample and make sure elements per buckets do not overflow
                        if data_split == "none" or i < days - 1:
                            # choose bucket
                            p = rand_u[k]
                            # retry of the bucket is full
                            while total_counter[p] + counter[p] >= total_per_file[p]:
                                p = np.random.randint(low=0, high=days_to_sample)
                        else:  # preserve the last day/bucket if needed
                            p = i
                        buckets[p].append(k)
                        counter[p] += 1
                else:  # randomize is day or none
                    for k in range(size):
                        # do not sample, preserve the data in this bucket
                        p = i
                        buckets[p].append(k)
                        counter[p] += 1

                # sanity check
                if np.sum(counter) != size:
                    sys.exit("ERROR: sanity check on number of samples failed")
                # debug prints
                # logging.debug(counter)
                # logging.debug(str(np.sum(counter)) + " = " + str(size))
                # logging.debug([len(x) for x in buckets])
                # logging.debug(total_counter)

                # partially feel the buckets
                for j in range(days):
                    filename_j_y = npzfile + "_{0}_intermediate_y.npy".format(j)
                    filename_j_d = npzfile + "_{0}_intermediate_d.npy".format(j)
                    filename_j_s = npzfile + "_{0}_intermediate_s.npy".format(j)
                    start = total_counter[j]
                    end = total_counter[j] + counter[j]
                    # target buckets
                    fj_y = np.load(filename_j_y, mmap_mode='r+')
                    # logging.debug("start=" + str(start) + " end=" + str(end)
                    #       + " end - start=" + str(end - start) + " "
                    #       + str(fj_y[start:end].shape) + " "
                    #       + str(len(buckets[j])))
                    fj_y[start:end] = y[buckets[j]]
                    del fj_y
                    # dense buckets
                    fj_d = np.load(filename_j_d, mmap_mode='r+')
                    # logging.debug("start=" + str(start) + " end=" + str(end)
                    #       + " end - start=" + str(end - start) + " "
                    #       + str(fj_d[start:end, :].shape) + " "
                    #       + str(len(buckets[j])))
                    fj_d[start:end, :] = X_int[buckets[j], :]
                    del fj_d
                    # sparse buckets
                    fj_s = np.load(filename_j_s, mmap_mode='r+')
                    # logging.debug("start=" + str(start) + " end=" + str(end)
                    #       + " end - start=" + str(end - start) + " "
                    #       + str(fj_s[start:end, :].shape) + " "
                    #       + str(len(buckets[j])))
                    fj_s[start:end, :] = X_cat[buckets[j], :]
                    del fj_s
                    # update counters for next step
                    total_counter[j] += counter[j]

        # 2nd pass of FYR shuffle
        # check if data already exists
        for j in range(days):
            filename_j = npzfile + "_{0}_reordered.npz".format(j)
            if path.exists(filename_j):
                logging.debug("Using existing " + filename_j)
            else:
                recreate_flag = True
        # reorder within buckets
        if recreate_flag:
            for j in range(days):
                filename_j_y = npzfile + "_{0}_intermediate_y.npy".format(j)
                filename_j_d = npzfile + "_{0}_intermediate_d.npy".format(j)
                filename_j_s = npzfile + "_{0}_intermediate_s.npy".format(j)
                fj_y = np.load(filename_j_y)
                fj_d = np.load(filename_j_d)
                fj_s = np.load(filename_j_s)

                indices = range(total_per_file[j])
                if randomize == "day" or randomize == "total":
                    if data_split == "none" or j < days - 1:
                        indices = np.random.permutation(range(total_per_file[j]))

                filename_r = npzfile + "_{0}_reordered.npz".format(j)
                logging.debug("Reordering (2nd pass) " + filename_r)
                np.savez_compressed(
                    filename_r,
                    X_cat=fj_s[indices, :],
                    X_int=fj_d[indices, :],
                    y=fj_y[indices],
                )

    else:
        logging.debug("Concatenating multiple days into %s.npz file" % str(d_path + o_filename))

        # load and concatenate data
        for i in range(days):
            filename_i = npzfile + "_{0}_processed.npz".format(i)
            with np.load(filename_i) as data:
                if i == 0:
                    X_cat = data["X_cat"]
                    X_int = data["X_int"]
                    y = data["y"]
                else:
                    X_cat = np.concatenate((X_cat, data["X_cat"]))
                    X_int = np.concatenate((X_int, data["X_int"]))
                    y = np.concatenate((y, data["y"]))
            logging.debug("Loaded day:", i, "y = 1:", len(y[y == 1]), "y = 0:", len(y[y == 0]))

        with np.load(d_path + d_file + "_fea_count.npz") as data:
            counts = data["counts"]
        logging.debug("Loaded counts!")

        logging.debug("saving compressed dataset")
        np.savez_compressed(
            d_path + o_filename + ".npz",
            X_cat=X_cat,
            X_int=X_int,
            y=y,
            counts=counts,
        )

    return d_path + o_filename + ".npz"


def getCriteoAdData(
        datafile: str,
        o_filename: str,
        max_ind_range: int = -1,
        sub_sample_rate: int = 0.0,
        days: int = 7,
        data_split: str = 'train',
        randomize: str = 'total',
        criteo_kaggle: bool = True,
        memory_map: bool = False,
        dataset_multiprocessing: bool = False,
) -> str:
    """Passes through entire dataset and defines dictionaries for categorical
    features and determines the number of total categories.

    Inputs:
       datafile : path to downloaded raw data file
       o_filename (str): saves results under o_filename if filename is not ""

    Output:
      o_file (str): output file path
    """
    # split the datafile into path and filename
    lstr = datafile.split("/")
    d_path = "/".join(lstr[0:-1]) + "/"
    d_file = lstr[-1].split(".")[0] if criteo_kaggle else lstr[-1]
    npzfile = d_path + ((d_file + "_day") if criteo_kaggle else d_file)
    trafile = d_path + ((d_file + "_fea") if criteo_kaggle else "fea")

    # count number of datapoints in training set
    total_file = d_path + d_file + "_day_count.npz"
    if path.exists(total_file):
        with np.load(total_file) as data:
            total_per_file = list(data["total_per_file"])
        total_count = np.sum(total_per_file)
        logging.debug("Skipping counts per file (already exist)")
    else:
        total_count = 0
        total_per_file = []
        if criteo_kaggle:
            # WARNING: The raw data consists of a single train.txt file
            # Each line in the file is a sample, consisting of 13 continuous and
            # 26 categorical features (an extra space indicates that feature is
            # missing and will be interpreted as 0).
            if path.exists(datafile):
                logging.debug("Reading data from path=%s" % (datafile))
                with open(str(datafile)) as f:
                    for _ in f:
                        total_count += 1
                total_per_file.append(total_count)
                # reset total per file due to split
                num_data_per_split, extras = divmod(total_count, days)
                total_per_file = [num_data_per_split] * days
                for j in range(extras):
                    total_per_file[j] += 1
                # split into days (simplifies code later on)
                file_id = 0
                boundary = total_per_file[file_id]
                nf = open(npzfile + "_" + str(file_id), "w")
                with open(str(datafile)) as f:
                    for j, line in enumerate(f):
                        if j == boundary:
                            nf.close()
                            file_id += 1
                            nf = open(npzfile + "_" + str(file_id), "w")
                            boundary += total_per_file[file_id]
                        nf.write(line)
                nf.close()
            else:
                sys.exit(
                    "ERROR: Criteo Kaggle Display Ad Challenge Dataset path is invalid; please download from "
                    "https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset"
                )
        else:
            # WARNING: The raw data consist of day_0.gz,... ,day_23.gz text files
            # Each line in the file is a sample, consisting of 13 continuous and
            # 26 categorical features (an extra space indicates that feature is
            # missing and will be interpreted as 0).
            for i in range(days):
                datafile_i = datafile + "_" + str(i)  # + ".gz"
                if path.exists(str(datafile_i)):
                    logging.debug("Reading data from path=%s" % (str(datafile_i)))
                    # file day_<number>
                    total_per_file_count = 0
                    with open(str(datafile_i)) as f:
                        for _ in f:
                            total_per_file_count += 1
                    total_per_file.append(total_per_file_count)
                    total_count += total_per_file_count
                else:
                    sys.exit(
                        "ERROR: Criteo Terabyte Dataset path is invalid; please download "
                        "from https://labs.criteo.com/2013/12/download-terabyte-click-logs"
                    )

    # process a file worth of data and reinitialize data
    # note that a file main contain a single or multiple splits
    def process_one_file(
            datfile: Any,
            npzfile: Any,
            split: Any,
            num_data_in_split: Any,
            dataset_multiprocessing: Any,
            convertDictsDay: Optional[Any] = None,
            resultDay: Optional[Any] = None
    ) -> Union[None, int]:
        if dataset_multiprocessing:
            convertDicts_day = [{} for _ in range(26)]

        with open(str(datfile)) as f:
            y = np.zeros(num_data_in_split, dtype="i4")  # 4 byte int
            X_int = np.zeros((num_data_in_split, 13), dtype="i4")  # 4 byte int
            X_cat = np.zeros((num_data_in_split, 26), dtype="i4")  # 4 byte int
            if sub_sample_rate == 0.0:
                rand_u = 1.0
            else:
                rand_u = np.random.uniform(low=0.0, high=1.0, size=num_data_in_split)

            i = 0
            percent = 0
            for k, line in enumerate(f):
                # process a line (data point)
                line = line.split('\t')
                # set missing values to zero
                for j in range(len(line)):
                    if (line[j] == '') or (line[j] == '\n'):
                        line[j] = '0'
                # sub-sample data by dropping zero targets, if needed
                target = np.int32(line[0])
                if target == 0 and \
                   (rand_u if sub_sample_rate == 0.0 else rand_u[k]) < sub_sample_rate:
                    continue

                y[i] = target
                X_int[i] = np.array(line[1:14], dtype=np.int32)
                if max_ind_range > 0:
                    X_cat[i] = np.array(
                        list(map(lambda x: int(x, 16) % max_ind_range, line[14:])),
                        dtype=np.int32
                    )
                else:
                    X_cat[i] = np.array(
                        list(map(lambda x: int(x, 16), line[14:])),
                        dtype=np.int32
                    )

                # count uniques
                if dataset_multiprocessing:
                    for j in range(26):
                        convertDicts_day[j][X_cat[i][j]] = 1
                    # debug prints
                    if float(i) / num_data_in_split * 100 > percent + 1:
                        percent = int(float(i) / num_data_in_split * 100)
                        logging.debug(
                            "Load %d/%d (%d%%) Split: %d  Label True: %d  Stored: %d"
                            % (
                                i,
                                num_data_in_split,
                                percent,
                                split,
                                target,
                                y[i],
                            ),
                            end="\n",
                        )
                else:
                    for j in range(26):
                        convertDicts[j][X_cat[i][j]] = 1
                    # debug prints
                    logging.debug(
                        "Load %d/%d  Split: %d  Label True: %d  Stored: %d"
                        % (
                            i,
                            num_data_in_split,
                            split,
                            target,
                            y[i],
                        ),
                        end="\r",
                    )
                i += 1

            # store num_data_in_split samples or extras at the end of file
            # count uniques
            # X_cat_t  = np.transpose(X_cat)
            # for j in range(26):
            #     for x in X_cat_t[j,:]:
            #         convertDicts[j][x] = 1
            # store parsed
            filename_s = npzfile + "_{0}.npz".format(split)
            if path.exists(filename_s):
                logging.debug("\nSkip existing " + filename_s)
            else:
                np.savez_compressed(
                    filename_s,
                    X_int=X_int[0:i, :],
                    # X_cat=X_cat[0:i, :],
                    X_cat_t=np.transpose(X_cat[0:i, :]),  # transpose of the data
                    y=y[0:i],
                )
                logging.debug("\nSaved " + npzfile + "_{0}.npz!".format(split))

        if dataset_multiprocessing:
            resultDay[split] = i
            convertDictsDay[split] = convertDicts_day
            return
        else:
            return i

    # create all splits (reuse existing files if possible)
    recreate_flag = False
    convertDicts = [{} for _ in range(26)]
    # WARNING: to get reproducable sub-sampling results you must reset the seed below
    # np.random.seed(123)
    # in this case there is a single split in each day
    for i in range(days):
        npzfile_i = npzfile + "_{0}.npz".format(i)
        npzfile_p = npzfile + "_{0}_processed.npz".format(i)
        if path.exists(npzfile_i):
            logging.debug("Skip existing " + npzfile_i)
        elif path.exists(npzfile_p):
            logging.debug("Skip existing " + npzfile_p)
        else:
            logging.debug("setting recreate for day", i)
            recreate_flag = True

    if recreate_flag:
        if dataset_multiprocessing:
            resultDay = Manager().dict()
            convertDictsDay = Manager().dict()
            processes = [Process(target=process_one_file,
                                 name="process_one_file:%i" % i,
                                 args=(npzfile + "_{0}".format(i),
                                       npzfile,
                                       i,
                                       total_per_file[i],
                                       dataset_multiprocessing,
                                       convertDictsDay,
                                       resultDay,
                                       )
                                 ) for i in range(0, days)]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            for day in range(days):
                total_per_file[day] = resultDay[day]
                logging.debug("Constructing convertDicts Split: {}".format(day))
                convertDicts_tmp = convertDictsDay[day]
                for i in range(26):
                    for j in convertDicts_tmp[i]:
                        convertDicts[i][j] = 1
        else:
            for i in range(days):
                total_per_file[i] = process_one_file(
                    npzfile + "_{0}".format(i),
                    npzfile,
                    i,
                    total_per_file[i],
                    dataset_multiprocessing,
                )

    # report and save total into a file
    total_count = np.sum(total_per_file)
    if not path.exists(total_file):
        np.savez_compressed(total_file, total_per_file=total_per_file)
    logging.debug("Total number of samples:", total_count)
    logging.debug("Divided into days/splits:\n", total_per_file)

    # dictionary files
    counts = np.zeros(26, dtype=np.int32)
    if recreate_flag:
        # create dictionaries
        for j in range(26):
            for i, x in enumerate(convertDicts[j]):
                convertDicts[j][x] = i
            dict_file_j = d_path + d_file + "_fea_dict_{0}.npz".format(j)
            if not path.exists(dict_file_j):
                np.savez_compressed(
                    dict_file_j,
                    unique=np.array(list(convertDicts[j]), dtype=np.int32)
                )
            counts[j] = len(convertDicts[j])
        # store (uniques and) counts
        count_file = d_path + d_file + "_fea_count.npz"
        if not path.exists(count_file):
            np.savez_compressed(count_file, counts=counts)
    else:
        # create dictionaries (from existing files)
        for j in range(26):
            with np.load(d_path + d_file + "_fea_dict_{0}.npz".format(j)) as data:
                unique = data["unique"]
            for i, x in enumerate(unique):
                convertDicts[j][x] = i
        # load (uniques and) counts
        with np.load(d_path + d_file + "_fea_count.npz") as data:
            counts = data["counts"]

    # process all splits
    if dataset_multiprocessing:
        processes = [
            Process(
                target=processCriteoAdData,
                name="processCriteoAdData:%i" % i,
                args=(d_path, d_file, npzfile, i, convertDicts, counts,)
            ) for i in range(0, days)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    else:
        for i in range(days):
            processCriteoAdData(d_path, d_file, npzfile, i, convertDicts, counts)

    output_path = Path(d_path + o_filename + ".npz")

    if not output_path.exists():
        o_file = concatCriteoAdData(
            d_path,
            d_file,
            npzfile,
            trafile,
            days,
            data_split,
            randomize,
            total_per_file,
            total_count,
            memory_map,
            o_filename
        )
    else:
        o_file = str(output_path)

    return o_file
