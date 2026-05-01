#
# Copyright (c) 2026
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
import os
import sys
import logging
import numpy as np
import pandas as pd
from ts_datasets.ts_datasets.anomaly.base import TSADBaseDataset

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.DEBUG)
_logger.addHandler(_handler)


class ACK(TSADBaseDataset):
    """
    AIClusterKPI dataset loader for the local `data/kad` folder.

    Expected files in `data/kad`:
      - KPI_31D_20260331--20260408.csv
      - KPI_31D_20260408--20260413.csv
      - KPI_31D_20260415--20260417_label.csv

    The first two files are treated as training data, and the labeled
    0417 file is treated as test data with anomaly labels.
    """

    filename = "ACK Dataset"
    fdir = os.path.dirname(os.path.abspath(__file__))
    merlion_root = os.path.abspath(os.path.join(fdir, "..", "..", ".."))
    data_dir = os.path.join(merlion_root, "data", "ack")
    valid_subsets = ["all"]

    def __init__(self, subset="all", rootdir=None):
        super().__init__()
        if subset == "all":
            subset = self.valid_subsets
        elif isinstance(subset, str):
            assert subset in self.valid_subsets, f"subset should be in {self.valid_subsets}, but got {subset}"
            subset = [subset]

        if rootdir is None:
            rootdir = ACK.data_dir

        for s in subset:
            df, metadata = combine_train_test_datasets(*ACK._load_data(directory=os.path.join(rootdir)))
            self.time_series.append(df)
            self.metadata.append(metadata)

    @staticmethod
    def _load_data(directory):
        processed_dir = os.path.join(directory)

        # ACK training files are fixed and should be loaded explicitly.
        train_paths = [
            os.path.join(processed_dir, "train1.csv"),
            os.path.join(processed_dir, "train2.csv"),
        ]
        test_path = os.path.join(processed_dir, "test.csv")

        missing = [p for p in train_paths + [test_path] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"Missing ACK files: {missing}")

        train_dfs = [pd.read_csv(path) for path in train_paths]

        train_df = pd.concat(train_dfs, axis=0, ignore_index=True)
        test_df = pd.read_csv(test_path)

        if "label" not in test_df.columns:
            raise ValueError(f"ACK label column not found in {test_path}")

        test_labels = np.array(test_df["label"]).astype(int)

        # Drop columns that are not features from test set
        drop_cols = [c for c in ["label", "fault_type", "target_node", "anomaly_details","event_tag", "event_id", "event_desc"] if c in test_df.columns]
        test_df = test_df.drop(columns=drop_cols)

        # train_df = train_df.drop(columns=drop_cols)

        return train_df, test_df, test_labels


def combine_train_test_datasets(train_df, test_df, test_labels):
    train_df.columns = [str(c) for c in train_df.columns]
    test_df.columns = [str(c) for c in test_df.columns]
    df = pd.concat([train_df, test_df]).reset_index(drop=True)

    if "timestamp" in df.columns:
        df.index = pd.to_datetime(df["timestamp"])
        df = df.drop(columns=["timestamp"])
    else:
        df.index = pd.RangeIndex(start=0, stop=df.shape[0], step=1)

    df.index.rename("timestamp", inplace=True)
    df.sort_index(inplace=True)

    if not df.index.is_monotonic_increasing:
        raise ValueError("KAD combined dataframe index is not monotonic increasing after sorting.")

    metadata = pd.DataFrame(
        {
            "trainval": df.index < df.index[train_df.shape[0]],
            "anomaly": np.concatenate([np.zeros(train_df.shape[0], dtype=int), test_labels]),
        },
        index=df.index,
    )
    return df, metadata
