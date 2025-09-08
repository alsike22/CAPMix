import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from dataloader.augmentations import DataTransform
from dataloader.generate_negative import *
from sklearn.model_selection import train_test_split
from utils import subsequences, calculate_fid, calculate_mahalanobis
from merlion.transform.normalize import MeanVarNormalize, MinMaxNormalize
from merlion.utils import TimeSeries
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
import fastdtw


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if hasattr(config, 'augmentation'):
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if hasattr(self, 'aug1'):
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(train_data, test_data, train_labels, test_labels, seed, configs):
    test_anomaly_window_num = int(len(np.where(test_labels[1:] != test_labels[:-1])[0]) / 2)

    train_x = subsequences(train_data, configs.window_size, configs.time_step)
    test_x = subsequences(test_data, configs.window_size, configs.time_step)
    train_y = subsequences(train_labels, configs.window_size, configs.time_step)
    test_y = subsequences(test_labels, configs.window_size, configs.time_step)

    train_y_window = np.zeros(train_x.shape[0])
    test_y_window = np.zeros(test_x.shape[0])
    train_anomaly_window_num = 0
    for i, item in enumerate(train_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            train_anomaly_window_num += 1
            train_y_window[i] = 1
        else:
            train_y_window[i] = 0
    for i, item in enumerate(test_y[:]):
        if sum(item[:configs.time_step]) >= 1:
            test_y_window[i] = 1
        else:
            test_y_window[i] = 0
    train_y = train_y_window
    test_y = test_y_window
    _, val_x, _, val_y = train_test_split(test_x, test_y_window, test_size=0.2, shuffle=True, random_state=seed,
                                          stratify=test_y_window)
    
    if configs.ab_cap:
    # 改了这个部分
        train_origin = train_x.copy()

        if ((configs.rate != 0) and (configs.rate <= 1)):
            train_aug_x = cut_add_paste_outlier_plus(train_origin, configs)
            sample_num = int(configs.rate * len(train_origin))
            sample_list = [i for i in range(sample_num)]
            sample_list = random.sample(sample_list, sample_num)
            sample = train_aug_x[sample_list, :, :]
        elif configs.rate > 1:
            train_aug_x_1 = cut_add_paste_outlier_plus(train_origin, configs)
            train_aug_x_2 = cut_add_paste_outlier_plus(train_origin, configs)
            train_aug_x = np.concatenate((train_aug_x_1, train_aug_x_2), axis=0)
            sample_num = int(configs.rate * len(train_origin))
            sample_list = [i for i in range(sample_num)]
            sample_list = random.sample(sample_list, sample_num)
            sample = train_aug_x[sample_list, :, :]
        else:
            sample_num = 0
            sample = train_x[[], :, :]

        train_aug_x = sample
        train_normal_index = np.where(train_y == 0)

        train_normal_x = train_x[train_normal_index]
        train_normal_y = np.zeros(len(train_normal_x))


    center_normal = np.sum(train_normal_x, axis=0) / train_normal_x.shape[0]
    all_distance = []
    for i, x_i in enumerate(sample):
        distance, path = fastdtw.fastdtw(center_normal, x_i)
        all_distance.append(distance)
    all_distance = np.array(all_distance)
    distance_mean = np.mean(all_distance)
    distance_std = np.std(all_distance)
    gamma = configs.gamma
  
    condition = all_distance < (distance_mean - gamma * distance_std)
    revise_aug_nor_index = np.where(condition)
    revise_aug_ano_index = np.where(~condition)

    revise_aug_nor_x = train_aug_x[revise_aug_nor_index]
    revise_aug_ano_x = train_aug_x[revise_aug_ano_index]


    revise_aug_nor_y = np.zeros(len(revise_aug_nor_x)) + 1 / gamma
    revise_aug_ano_y = np.zeros(len(revise_aug_ano_x)) + 1
    revise_aug_x = np.concatenate((revise_aug_nor_x, revise_aug_ano_x), axis=0)
    revise_aug_y = np.concatenate((revise_aug_nor_y, revise_aug_ano_y), axis=0)
    train_x = np.concatenate((train_x, revise_aug_x), axis=0)
    train_y = np.concatenate((train_y, revise_aug_y), axis=0)


    alpha = configs.alpha
    if alpha > 0 and configs.layer_mix == 0:
        mix_num = int(alpha * len(revise_aug_ano_x))
        mix_list = random.sample(list(revise_aug_ano_x), mix_num)
        to_mix_index_list = np.random.choice(range(len(train_normal_x)), size=mix_num, replace=True)
        to_mix_x = train_normal_x[to_mix_index_list]
        to_mix_y = train_normal_y[to_mix_index_list]

        lams = np.random.beta(alpha, alpha, mix_num)
        mixed_y = lams * ((np.zeros(mix_num) + 1)) + (1 - lams) * to_mix_y
        lams = lams.reshape(mix_num, 1, 1)
        mixed_x = lams * np.array(mix_list) + (1 - lams) * np.array(to_mix_x)

        train_x = np.concatenate((train_x, mixed_x), axis=0)
        train_y = np.concatenate((train_y, mixed_y), axis=0)

    
    num_samples = len(train_x)
    batches = int(num_samples / configs.batch_size)

    if num_samples % configs.batch_size <= 1:
        train_x = train_x[0:batches * configs.batch_size - 1]  
        train_y = train_y[0:batches * configs.batch_size - 1]

    train_x = train_x.transpose((0, 2, 1))
    val_x = val_x.transpose((0, 2, 1))
    test_x = test_x.transpose((0, 2, 1))

    train_dat_dict = dict()
    train_dat_dict["samples"] = train_x
    train_dat_dict["labels"] = train_y

    val_dat_dict = dict()
    val_dat_dict["samples"] = val_x
    val_dat_dict["labels"] = val_y

    test_dat_dict = dict()
    test_dat_dict["samples"] = test_x
    test_dat_dict["labels"] = test_y

    train_dataset = Load_Dataset(train_dat_dict, configs)
    val_dataset = Load_Dataset(val_dat_dict, configs)
    test_dataset = Load_Dataset(test_dat_dict, configs)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                             shuffle=True, drop_last=False,
                                             num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.batch_size,
                                              shuffle=False, drop_last=False,
                                              num_workers=0)
    return train_loader, val_loader, test_loader, test_anomaly_window_num

