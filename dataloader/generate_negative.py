import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from utils import calculate_mahalanobis_of_samples
import fastdtw


def sine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    # timestamp = np.linspace(0, 10, length)
    timestamp = np.arange(length)
    value = np.sin(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        noise = np.random.normal(0, 1, length)
        value = value + noise_amp * noise
    value = coef * value + offset
    return value


def square_sine(level=5, length=500, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    value = np.zeros(length)
    for i in range(level):
        value += 1 / (2 * i + 1) * sine(length=length, freq=freq * (2 * i + 1), coef=coef, offset=offset, noise_amp=noise_amp)
    return value


# Add collective point outliers to original data (Variant)
def point_outliers(train_x, configs):
    for i, x_i in enumerate(train_x):
        if x_i.shape[1] > 1:
            elements = np.arange(0, x_i.shape[1])
            if configs.dim > 1:
                dim_size = np.random.randint(1, configs.dim)
            else:
                dim_size = 1
            selected_elements = np.random.choice(elements, size=dim_size, replace=False)
            for item in selected_elements:
                position = int(np.random.rand() * train_x.shape[1])
                local_std = x_i[:, item].std()
                local_mean = x_i[:, item].mean()
                scale = local_std * np.random.choice((-1, 1)) * 3 * (np.random.rand() + 1)
                point_value = local_mean + scale
                train_x[i, position, item] = point_value
        else:
            position = int(np.random.rand() * train_x.shape[1])
            local_std = x_i[:, 0].std()
            local_mean = x_i[:, 0].mean()
            scale = local_std * np.random.choice((-1, 1)) * 3 * (np.random.rand()+1)
            point_value = local_mean + scale
            train_x[i, position, 0] = point_value
    return train_x


# Add collective trend outliers to original data (Variant)
def collective_trend_outliers(train_x, configs):
    for i, x_i in enumerate(train_x):
        if x_i.shape[1] > 1:
            elements = np.arange(0, x_i.shape[1])
            if configs.dim > 1:
                dim_size = np.random.randint(1, configs.dim)
            else:
                dim_size = 1
            selected_elements = np.random.choice(elements, size=dim_size, replace=False)
            for item in selected_elements:
                radius = max(int(train_x.shape[1]/6), int(np.random.rand() * train_x.shape[1]))
                factor = np.random.rand() * configs.trend_rate
                position = int(np.random.rand() * (train_x.shape[1] - radius))
                start, end = position, position + radius
                slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
                train_x[i, start:end, item] = x_i[start:end, item] + slope
        else:
            radius = max(int(train_x.shape[1] / 6), int(np.random.rand() * train_x.shape[1]))
            factor = np.random.rand() * configs.trend_rate
            position = int(np.random.rand() * (train_x.shape[1] - radius))
            start, end = position, position + radius
            slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
            train_x[i, start:end, 0] = x_i[start:end, 0] + slope
    return train_x


# Add collective seasonal outliers to original data
def collective_seasonal_outliers(train_x):
    seasonal_config = {'length': 400, 'freq': 0.04, 'coef': 1.5, "offset": 0.0, 'noise_amp': 0.05}
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/6), int(np.random.rand() * train_x.shape[1]))
        factor = np.random.rand()
        seasonal_config['freq'] = factor * seasonal_config['freq']
        position = int(np.random.rand() * (train_x.shape[1] - radius))
        start, end = position, position + radius
        train_x[i, start:end, 0] = sine(**seasonal_config)[start:end]
    return train_x


# Add cut outliers to original data (Variant)
def cut_outliers(train_x):
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/6), int(np.random.rand() * (train_x.shape[1]-2)))
        to_position = int(random.uniform(0, train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = 0
    return train_x


# From the same sequence, Add outliers to original data via the CutPaste method (Variant)
def cut_paste_outliers_same(train_x):
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/6), int(np.random.rand() * (train_x.shape[1]-2)))
        cut_data = x_i
        position = random.sample(range(0, train_x.shape[1] - radius + 1), 2)
        from_position = position[0]
        to_position = position[1]
        cut_data = cut_data[from_position:from_position + radius, :]
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]
    return train_x


# From different sequences, Add outliers to original data via the CutPaste method (Variant)
def cut_paste_outliers_other(train_x):
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/6), int(np.random.rand() * (train_x.shape[1]-2)))
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        from_position = int(random.uniform(0, train_x.shape[1] - radius))
        to_position = int(random.uniform(0, train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = cut_data[from_position:from_position + radius, :]
    return train_x


# For multidimensional time series data, add the same trend to each dimension (Variant)
# CutAddPaste with same trends (SameCAP)
def cut_paste_outliers_same_trend(train_x, configs):
    for i, x_i in enumerate(train_x):
        radius = max(int(train_x.shape[1]/3), int(np.random.rand() * train_x.shape[1]))
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        from_position = int(np.random.rand() * (train_x.shape[1] - radius))
        cut_data = cut_data[from_position:from_position + radius, :]
        factor = np.random.rand() * configs.trend_rate
        slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
        slope = slope.reshape(-1, 1)
        slope = np.tile(slope, configs.input_channels)
        cut_data[:, :] += slope
        to_position = int(np.random.rand() * (train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]
    return train_x


# Add outliers to original data via our CutAddPaste method
def cut_add_paste_outlier(train_x, configs):
    for i, x_i in enumerate(train_x):
        # radius = max(int(train_x.shape[1]/6), int(np.random.rand() * train_x.shape[1]))
        radius = max(int(configs.cut_rate), int(np.random.rand() * train_x.shape[1]))

        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        from_position = int(np.random.rand() * (train_x.shape[1] - radius))
        cut_data = cut_data[from_position:from_position + radius, :]
        if cut_data.shape[1] > 1:
            elements = np.arange(0, cut_data.shape[1])
            if configs.dim > 1:
                dim_size = np.random.randint(1, configs.dim)
            else:
                dim_size = 1
            selected_elements = np.random.choice(elements, size=dim_size, replace=False)
            for item in selected_elements:
                factor = np.random.rand() * configs.trend_rate
                slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
                cut_data[:, item] += slope
        else:
            factor = np.random.rand() * configs.trend_rate
            slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
            cut_data[:, 0] += slope
        to_position = int(np.random.rand() * (train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]
    return train_x

#皮尔逊系数sl
def cut_add_paste_outlier_with_sl(train_x, train_y, configs):
    for i, x_i in enumerate(train_x):
        # radius = max(int(train_x.shape[1]/6), int(np.random.rand() * train_x.shape[1]))
        radius = max(int(configs.cut_rate), int(np.random.rand() * train_x.shape[1]))

        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        x_j = cut_data
        x_oi = x_i.copy()
        from_position = int(np.random.rand() * (train_x.shape[1] - radius))
        cut_data = cut_data[from_position:from_position + radius, :]
        if cut_data.shape[1] > 1:
            elements = np.arange(0, cut_data.shape[1])
            if configs.dim > 1:
                dim_size = np.random.randint(1, configs.dim)
            else:
                dim_size = 1
            selected_elements = np.random.choice(elements, size=dim_size, replace=False)
            for item in selected_elements:
                factor = np.random.rand() * configs.trend_rate
                slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
                cut_data[:, item] += slope
        else:
            factor = np.random.rand() * configs.trend_rate
            slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
            cut_data[:, 0] += slope
        to_position = int(np.random.rand() * (train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]
        x_k = train_x[i,:,:]
        

        s_i,_ = (0,0) if (np.std(x_oi) == 0 or np.std(x_k) == 0) else pearsonr(x_oi.flatten(), x_k.flatten()) 
        s_j,_ = (0,0) if (np.std(x_j) == 0 or np.std(x_k) == 0) else pearsonr(x_j.flatten(), x_k.flatten())

        # s_i = cosine_similarity(x_oi, x_k)[0][0]
        # s_j = cosine_similarity(x_j, x_k)[0][0]
        s_i = 0 if np.isnan(s_i) else s_i
        s_j = 0 if np.isnan(s_j) else s_j

        y_k = (1-s_i) * (1-s_j)
        y_k = np.clip(y_k, 0, 1)

        train_y[i] = y_k

    return train_x, train_y


def cut_add_paste_outlier_with_mah(train_x, train_y, configs):
    for i, x_i in enumerate(train_x):
        # radius = max(int(train_x.shape[1]/6), int(np.random.rand() * train_x.shape[1]))
        radius = max(int(configs.cut_rate), int(np.random.rand() * train_x.shape[1]))

        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        x_j = cut_data
        x_oi = x_i.copy()
        from_position = int(np.random.rand() * (train_x.shape[1] - radius))
        cut_data = cut_data[from_position:from_position + radius, :]
        if cut_data.shape[1] > 1:
            elements = np.arange(0, cut_data.shape[1])
            if configs.dim > 1:
                dim_size = np.random.randint(1, configs.dim)
            else:
                dim_size = 1
            selected_elements = np.random.choice(elements, size=dim_size, replace=False)
            for item in selected_elements:
                factor = np.random.rand() * configs.trend_rate
                slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
                cut_data[:, item] += slope
        else:
            factor = np.random.rand() * configs.trend_rate
            slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
            cut_data[:, 0] += slope
        to_position = int(np.random.rand() * (train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]
        x_k = train_x[i,:,:]
        
        normal_radius = 2*calculate_mahalanobis_of_samples(x_oi, x_j)
        dis1 = calculate_mahalanobis_of_samples(x_k, x_oi)
        dis2 = calculate_mahalanobis_of_samples(x_k, x_j)
        # cos_angle = (dis1**2 + dis2**2 - (normal_radius)**2) / (2 * dis1 * dis2)
        y_k = (dis1 + dis2) / normal_radius 
        train_y[i] = y_k
        # k = 1000
        # train_y[i] = 0.5 + 0.5 / (1 + np.exp(-k * (y_k - 1)))
        # print(train_y[i])

    return train_x, train_y


def cut_add_paste_outlier_plus(train_x, configs):
    # train_x_copy = train_x.copy()
    # all_distance_cut, all_distance_paste = [], []
    for i, x_i in enumerate(train_x):
        # radius = max(int(train_x.shape[1]/6), int(np.random.rand() * train_x.shape[1]))
        radius = max(int(configs.cut_rate), int(np.random.rand() * train_x.shape[1]))
        cut_random = int(random.uniform(0, train_x.shape[0]))
        cut_data = train_x[cut_random, :, :]
        # cut_data_copy = cut_data.copy()
        # paste_data_copy = train_x[cut_random, :, :].copy()
        from_position = int(np.random.rand() * (train_x.shape[1] - radius))
        cut_data = cut_data[from_position:from_position + radius, :]
        if cut_data.shape[1] > 1:
            elements = np.arange(0, cut_data.shape[1])
            if configs.dim > 1:
                dim_size = np.random.randint(1, configs.dim)
            else:
                dim_size = 1
            selected_elements = np.random.choice(elements, size=dim_size, replace=False)
            for item in selected_elements:
                factor = np.random.rand() * configs.trend_rate
                slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
                cut_data[:, item] += slope
        else:
            factor = np.random.rand() * configs.trend_rate
            slope = np.random.choice([-1, 1]) * factor * np.arange(radius)
            cut_data[:, 0] += slope
        to_position = int(np.random.rand() * (train_x.shape[1] - radius))
        train_x[i, to_position:to_position + radius, :] = cut_data[:, :]
        # gene_data_copy = train_x[i, :, :].copy()

        # distance_cut, path_cut = fastdtw.fastdtw(cut_data_copy, gene_data_copy)
        # distance_paste, path_paste = fastdtw.fastdtw(paste_data_copy, gene_data_copy)
        # all_distance_cut.append(distance_cut)
        # all_distance_paste.append(distance_paste)
    # all_distance_cut = np.array(all_distance_cut)
    # all_distance_paste = np.array(all_distance_paste)
    # print("all_distance_cut shape:", all_distance_cut.shape)
    # print("all_distance_paste shape:", all_distance_paste.shape)
    # return train_x, all_distance_cut, all_distance_paste
    return train_x