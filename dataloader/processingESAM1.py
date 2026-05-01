import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from merlion.utils import TimeSeries
from merlion.transform.normalize import MeanVarNormalize
from glob import glob
from tqdm import tqdm
from dateutil.parser import parse as parse_date
import matplotlib.pyplot as plt

def esa():
    dataset_splits = {"3_months": "2000-04-01",
                  "10_months": "2000-11-01",
                  "21_months": "2001-10-01",
                  "42_months": "2003-07-01",
                  "84_months": "2007-01-01"}
    test_data_split = "2007-01-01"

    dataset_folder = os.path.join('data/', 'esa')
    labels_df = pd.read_csv(os.path.join(dataset_folder, "labels.csv"), parse_dates=["StartTime", "EndTime"], date_parser=lambda x: parse_date(x, ignoretz=True))
    anomaly_types_df = pd.read_csv(os.path.join(dataset_folder, "anomaly_types.csv"))
    print(dataset_folder)
    print(labels_df.head())

    extension = ".zip"

    params_dict = {}

    for param in tqdm([f"channel_{i}" for i in range(41,47)]):
        print(param)
        param_df = pd.read_pickle(os.path.join(dataset_folder, "channels", f"{param}{extension}"))
        param_df["label"] = np.uint8(0)
        param_df = param_df.rename(columns={param: "value"})
        print(param_df.shape)
        print(param_df.head()) 
        if param_df["value"].dtype == "O":
                    print(f"{param} is not numeric!")
                    param_df["value"] = pd.factorize(param_df["value"])[0]
        
        is_param_annotated = False
        for _, row in labels_df.iterrows():
            if row["Channel"] == param:
                anomaly_type = anomaly_types_df.loc[anomaly_types_df["ID"] == row["ID"]]["Category"].values[0]
                if anomaly_type == "Anomaly":
                    label_value = 1 #Anomaly
                elif anomaly_type == "Rare Event":
                    label_value = 0#Rare Event #我把这个当正常
                param_df.loc[row["StartTime"]:row["EndTime"], "label"] = label_value
                is_param_annotated = True


        # param_df = param_df[param_df.index > parse_date(test_data_split)].copy()

        if len(param_df) == 0:
            params_dict[param] = []
            continue

        # Resample using zero order hold
        resampling_rule = pd.Timedelta(seconds=18)
        first_index_resampled = pd.Timestamp(param_df.index[0]).floor(freq=resampling_rule)
        last_index_resampled = pd.Timestamp(param_df.index[-1]).ceil(freq=resampling_rule)
        resampled_range = pd.date_range(first_index_resampled, last_index_resampled, freq=resampling_rule)
        params_dict[param] = param_df.reindex(resampled_range, method="ffill")
        params_dict[param].iloc[0] = param_df.iloc[0]  # Initialize the first sample

        # Restore annotated samples if not present in the resampled series
        if is_param_annotated:
            grouper = param_df.groupby(pd.Grouper(freq=resampling_rule))
            for timestamp, group in grouper.indices.items():
                if len(group) <= 1:
                    continue
                org_elements = param_df.iloc[group]
                if org_elements.label.values[-1] != 0: #normal
                    continue
                is_annotated = (org_elements.label > 0)
                if is_annotated.any():
                    print(timestamp, org_elements[is_annotated].iloc[-1])
                    params_dict[param].loc[timestamp + pd.Timedelta(resampling_rule)] = org_elements[is_annotated].iloc[-1]

    start_time, end_time = find_full_time_range(params_dict)
    full_index = pd.date_range(start_time, end_time, freq=resampling_rule)
    data_df = pd.DataFrame(index=full_index)

    all_params = list(params_dict.keys())
    for param in all_params:
        df = params_dict.pop(param)
        if len(df) == 0:
            data_df[param] = np.uint8(0)
            data_df[f"is_anomaly_{param}"] = np.uint8(0)
            continue

        df = df.rename(columns={"value": param, "label": f"is_anomaly_{param}"})
        data_df[df.columns] = df.reindex(data_df.index)
        data_df[param] = data_df[param].astype(np.float64).ffill().bfill()
        data_df[f"is_anomaly_{param}"] = data_df[f"is_anomaly_{param}"].ffill().bfill().astype(np.uint8)

    anomaly_columns = [col for col in data_df.columns if col.startswith("is_anomaly_")]
    data_df["label"] = (data_df[anomaly_columns] == 1).any(axis=1).astype(np.uint8)
    print(data_df.head())
    print(anomaly_columns)
    # 删除原来的 is_anomaly_* 列
    data_df = data_df.drop(columns=anomaly_columns)

    visualize_all_dimensions(data_df, output_path=os.path.join(dataset_folder, "data_df_visualization.png"))

    train_param_df = data_df[data_df.index <= parse_date(test_data_split)].copy()
    test_param_df = data_df[data_df.index > parse_date(test_data_split)].copy()



    train_labels = train_param_df["label"]
    train_param_df.drop(columns=["label"], inplace=True)
    test_labels = test_param_df["label"]
    test_param_df.drop(columns=["label"], inplace=True)

    test_param_df = test_param_df.rename_axis("datetime")
    test_labels = test_labels.rename_axis("datetime")
    train_param_df = train_param_df.rename_axis("datetime")
    train_labels = train_labels.rename_axis("datetime")

    # test_labels.to_csv(os.path.join(dataset_folder, "esa_test_labels.csv"))
    # train_labels.to_csv(os.path.join(dataset_folder, "esa_train_labels.csv"))

    # train_param_df.to_csv(os.path.join(dataset_folder, "esa_train.csv"))
    # test_param_df.to_csv(os.path.join(dataset_folder, "esa_test.csv"))

    visualize_data(
        train_param_df,
        train_labels,
        output_path=os.path.join(dataset_folder, "esa_train_visualization.png")
    )

    # 可视化测试集
    visualize_data(
        test_param_df,
        test_labels,
        output_path=os.path.join(dataset_folder, "esa_test_visualization.png")
    )

    np.save(os.path.join(dataset_folder, "esa_train.npy"), train_param_df)
    np.save(os.path.join(dataset_folder, "esa_test.npy"), test_param_df)
    np.save(os.path.join(dataset_folder, "esa_test_labels.npy"), test_labels)


def find_full_time_range(params_dict: dict):
    # Find full dataset time range
    start_time = []
    end_time = []
    for df in params_dict.values():
        if len(df) == 0:
            continue
        start_time.append(df.index[0])
        end_time.append(df.index[-1])
    start_time = min(start_time)
    end_time = max(end_time)

    return start_time, end_time


def visualize_data(data_df, anomaly_labels, output_path="visualization.png"):
    """
    可视化数据集，绘制多维时间序列，并用红色标出异常值。

    :param data_df: 数据集的 DataFrame，包含多维时间序列。
    :param anomaly_labels: 异常标签列，1 表示异常，0 表示正常。
    :param output_path: 保存可视化结果的路径。
    """
    plt.figure(figsize=(15, 10))
    num_dimensions = min(11, data_df.shape[1])  # 限制最多绘制 11 维数据
    time_index = data_df.index

    for i, column in enumerate(data_df.columns[:num_dimensions]):
        plt.subplot(num_dimensions, 1, i + 1)
        plt.plot(time_index, data_df[column], label=column, color="blue", linewidth=0.5)
        plt.ylabel(column)
        plt.xlabel("Time")
        plt.grid(True)

        # 标出异常值
        anomaly_indices = anomaly_labels[anomaly_labels == 1].index
        for anomaly_time in anomaly_indices:
            plt.axvline(x=anomaly_time, color="red", linestyle="--", linewidth=0.8)

        if i == 0:
            plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


def visualize_all_dimensions(data_df, output_path="data_df_visualization.png"):
    """
    可视化 data_df 中的每个维度，绘制多维时间序列。

    :param data_df: 数据集的 DataFrame，包含多维时间序列。
    :param output_path: 保存可视化结果的路径。
    """
    plt.figure(figsize=(15, 10))
    num_dimensions = min(11, data_df.shape[1])  # 限制最多绘制 11 维数据
    time_index = data_df.index

    for i, column in enumerate(data_df.columns[:num_dimensions]):
        plt.subplot(num_dimensions, 1, i + 1)
        plt.plot(time_index, data_df[column], label=column, color="blue", linewidth=0.5)
        plt.ylabel(column)
        plt.xlabel("Time")
        plt.grid(True)
        if i == 0:
            plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    esa()