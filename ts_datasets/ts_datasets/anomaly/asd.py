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


class ASD(TSADBaseDataset):
    """
    The ASD dataset is collected from a large Internet company. It characterizes the status of diﬀerent
    servers (entities) using a group of metrics. A group of stable services are run on each server, thus
    the data does not experience service changes or concept drifts during the time period included in
    the dataset.
    ASD contains 12 diﬀerent entities (each for a server), each of which has 19 metrics characterizing
    the status of the server (including CPU-related metrics, memory-related metrics, network metrics,
    virtual machine metrics, etc.).
    The data points in ASD are equally-spaced 5 minutes apart. The first 30-day data are training set
    (the last 30% data in training set are kept for validation), while the last 15-day data are testing set.
    Anomalies and their most anomalous dimensions in ASD testing set have been labeled by domain
    experts based on incident reports and domain knowledge.

    - source: https://github.com/zhhlee/InterFusion/tree/main
    """

    filename = "Application Server Dataset"
    url = "https://github.com/zhhlee/InterFusion/tree/main/data/processed"
    fdir = os.path.dirname(os.path.abspath(__file__))
    merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
    data_dir = os.path.join(merlion_root, "data", "asd")
    valid_subsets = sorted([
        f.split("_train.pkl")[0]  
        for f in os.listdir(data_dir)
        if f.endswith("_train.pkl")
    ])

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
            rootdir = os.path.join(merlion_root, "data", "asd")


        for s in subset:
            # Load training/test datasets
            df, metadata = combine_train_test_datasets(
                *ASD._load_data(directory=os.path.join(rootdir), sequence_name=s)
            )
            self.time_series.append(df)
            self.metadata.append(metadata)

    @staticmethod
    def _load_data(directory, sequence_name):

        processed_dir = os.path.join(directory)

        # print(f"Loading ASD dataset from {processed_dir} for sequence {sequence_name}...")
        
        train_path = os.path.join(processed_dir, f"{sequence_name}_train.pkl")
        test_path = os.path.join(processed_dir, f"{sequence_name}_test.pkl")
        label_path = os.path.join(processed_dir, f"{sequence_name}_test_label.pkl")

        with open(train_path, "rb") as f:
            train_data = pickle.load(f)
        with open(test_path, "rb") as f:
            test_data = pickle.load(f)
        with open(label_path, "rb") as f:
            test_labels = pickle.load(f)

        # test_labels[test_labels > 0] = 1
        # print(np.sum(test_labels[test_labels != 0]), np.sum(test_labels[test_labels == 1]))
        
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

