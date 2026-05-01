#
# Copyright (c) 2021 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import os
import sys
import logging
import requests
import tarfile
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from ts_datasets.ts_datasets.anomaly.base import TSADBaseDataset

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.DEBUG)
_logger.addHandler(_handler)


class Exathlon(TSADBaseDataset):
    """
    Implements the Exathlon dataset from [Jacob2021]_.
    The data was collected by running different applications on a Spark cluster and recording metrics from the Spark
    service and the worker nodes. We consider the trace for each app a separate dataset. You can control which app trace
    to load by setting the `app_id` parameter.

    - source: https://github.com/exathlonbenchmark/exathlon.git
    """

    filename = "exathlon"
    url = "https://github.com/exathlonbenchmark/exathlon.git"
    fdir = os.path.dirname(os.path.abspath(__file__))
    merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
    data_dir = os.path.join(merlion_root, "data", "exathlon")
    valid_subsets = sorted({f"app{str(i).zfill(2)}" for i in list(range(1, 7)) + [9, 10]})

    def __init__(self, subset="all", rootdir=None):
        super().__init__()
        if subset == "all":
            subset = self.valid_subsets
        elif type(subset) == str:
            assert subset in self.valid_subsets, f"subset should be in {self.valid_subsets}, but got {subset}"
            subset = [subset]

        if rootdir is None:
            fdir = os.path.dirname(os.path.abspath(__file__))
            merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
            rootdir = os.path.join(merlion_root, "data", "exathlon")


        for s in subset:
            # Load training/test datasets
            df, metadata = combine_train_test_datasets(
                *Exathlon._load_data(directory=os.path.join(rootdir), sequence_name=s)
            )
            self.time_series.append(df)
            self.metadata.append(metadata)

    @staticmethod
    def _load_data(directory, sequence_name):
        processed_dir = os.path.join(directory, sequence_name)

        train_path = os.path.join(processed_dir, "train.npy")
        test_path = os.path.join(processed_dir, "test.npy")
        label_path = os.path.join(processed_dir, "y_test.npy")

        train_data = np.load(train_path, allow_pickle=True)
        test_data = np.load(test_path, allow_pickle=True)
        test_labels = np.load(label_path, allow_pickle=True)

        train_data = np.concatenate(train_data, axis=0)
        test_data = np.concatenate(test_data, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        test_labels[test_labels > 0] = 1  # Binarize the labels
        
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)
        test_labels = np.array(test_labels).astype(int)

        return train_df, test_df, test_labels


def combine_train_test_datasets(train_df, test_df, test_labels):
    train_df.columns = [str(c) for c in train_df.columns]
    test_df.columns = [str(c) for c in test_df.columns]
    df = pd.concat([train_df, test_df]).reset_index()
    if "index" in df:
        df.drop(columns=["index"], inplace=True)
    df.index = pd.to_datetime(df.index * 60, unit="s")
    df.index.rename("timestamp", inplace=True)
    # There are no labels for training examples, so the training labels are set to 0 by default
    # The dataset is only for unsupervised time series anomaly detection
    metadata = pd.DataFrame(
        {
            "trainval": df.index < df.index[train_df.shape[0]],
            "anomaly": np.concatenate([np.zeros(train_df.shape[0], dtype=int), test_labels]),
        },
        index=df.index,
    )
    return df, metadata
