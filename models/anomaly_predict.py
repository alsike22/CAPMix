from affiliation.generics import convert_vector_to_events
from affiliation.metrics import pr_from_events
import numpy as np
import pandas as pd
from merlion.evaluate.anomaly import accumulate_tsad_score, ScoreType
from merlion.utils import TimeSeries
import time
from joblib import Parallel, delayed


# Anomalies detection
def ad_predict(target, scores, mode, nu):
    if_aff = np.count_nonzero(target)
    scores = np.array(scores)
    assert mode in ['direct','one-anomaly','fix','floating'], f"Invalid value: Must be one of modes"
    if mode == 'direct':
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_direct(target, scores, if_aff)
    elif mode == 'one-anomaly':
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_one_anomaly(target, scores, if_aff)
    elif mode == 'fix':
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_fix(target, scores, if_aff, nu)
    else:
        # affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_floating_lazy(target, scores, if_aff)
        affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_floating(target, scores, if_aff)
        # affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict = ad_floating_origin(target, scores, if_aff,1000)

    return affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict


# Anomalies are detected directly based on classification probabilities
def ad_direct(target, predict, if_aff):
    predict = np.int64(predict > 0.5)
    if if_aff != 0:
        events_gt = convert_vector_to_events(target)
        events_pred = convert_vector_to_events(predict)
        Trange = (0, len(predict))
        affiliation = pr_from_events(events_pred, events_gt, Trange)
    else:
        affiliation = dict()
        affiliation["precision"] = 0
        affiliation["recall"] = 0
    target_ts = TimeSeries.from_pd(pd.DataFrame(target))
    predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
    score = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
    return affiliation, score, score, score, predict


# For UCR dataset, there is only one anomaly period in the test set.
# Anomalies are detected based on maximum anomaly scores.
def ad_one_anomaly(target, scores, if_aff):
    scores = z_score(scores)
    threshold = np.max(scores, axis=0)
    max_number = np.sum(scores == threshold)
    predict = np.zeros(len(scores))
    # Prevents some methods from generating too many maximum values for anomaly scores.
    if max_number <= 10:
        for index, r2 in enumerate(scores):
            if r2.item() >= threshold:
                predict[index] = 1
    if if_aff != 0:
        events_gt = convert_vector_to_events(target)
        events_pred = convert_vector_to_events(predict)
        Trange = (0, len(predict))
        affiliation_max = pr_from_events(events_pred, events_gt, Trange)
    else:
        affiliation_max = dict()
        affiliation_max["precision"] = 0
        affiliation_max["recall"] = 0
    target_ts = TimeSeries.from_pd(pd.DataFrame(target))
    predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
    score_max = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
    return affiliation_max, score_max, score_max, score_max, predict


# Anomalies are detected based on anomaly scores and fixed thresholds.
def ad_fix(target, scores, if_aff, nu):
    scores = z_score(scores)
    detect_nu = 100 * (1 - nu)
    threshold = np.percentile(scores, detect_nu)
    predict = np.int64(scores > threshold)
    if if_aff != 0:
        events_gt = convert_vector_to_events(target)
        events_pred = convert_vector_to_events(predict)
        Trange = (0, len(predict))
        affiliation_max = pr_from_events(events_pred, events_gt, Trange)
    else:
        affiliation_max = dict()
        affiliation_max["precision"] = 0
        affiliation_max["recall"] = 0
    target_ts = TimeSeries.from_pd(pd.DataFrame(target))
    predict_ts = TimeSeries.from_pd(pd.DataFrame(predict))
    score_max = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
    return affiliation_max, score_max, score_max, score_max, predict


# Through Z-score, anomalies are detected based on anomaly scores and floating thresholds.
def ad_floating(target, scores, if_aff):
    start_time = time.time()

    normal_scores = z_score(scores)
    events_gt = convert_vector_to_events(target)
    target_ts = create_time_series(target, chunk_size=100000)
    # target_ts = TimeSeries.from_pd(pd.DataFrame(target))

    nu_list = np.arange(-3, 3, 0.01)
    pa_f1_list, pw_f1_list, rpa_f1_list = [], [], []
    affiliation_f1_list, affiliation_list = [], []
    score_list = []

    def _calc_f1_for_nu(detect_nu):
        predict = np.int64(normal_scores > detect_nu)
        if if_aff != 0:
            events_pred = convert_vector_to_events(predict)
            Trange = (0, len(predict))
            dic = pr_from_events(events_pred, events_gt, Trange)
            affiliation_f1 = 2 * (dic["precision"] * dic["recall"]) / (dic["precision"] + dic["recall"])
        else:
            dic = {"precision": 0, "recall": 0}
            affiliation_f1 = 0
        predict_ts = create_time_series(predict, chunk_size=100000)
        score = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
        rpa_f1 = score.f1(ScoreType.RevisedPointAdjusted)
        pa_f1 = score.f1(ScoreType.PointAdjusted)
        pw_f1 = score.f1(ScoreType.Pointwise)
        return dic, affiliation_f1, rpa_f1, pa_f1, pw_f1, score

    # results = Parallel(n_jobs=-1)(
    results = Parallel(n_jobs=4)(
        delayed(_calc_f1_for_nu)(detect_nu) for detect_nu in nu_list
    )

    for item in results:
        dic, aff_f1, rpa_f1, pa_f1, pw_f1, sc = item
        affiliation_f1_list.append(aff_f1)
        affiliation_list.append(dic)
        rpa_f1_list.append(rpa_f1)
        pa_f1_list.append(pa_f1)
        pw_f1_list.append(pw_f1)
        score_list.append(sc)

    affiliation_f1_list = np.nan_to_num(affiliation_f1_list)
    affiliation_max_index = np.nanargmax(affiliation_f1_list, axis=0)
    affiliation_max = affiliation_list[affiliation_max_index]
    affiliation_nu_max = nu_list[affiliation_max_index]
    print("Best affiliation threshold: {:.3f}".format(affiliation_nu_max))

    rpa_max_index = np.nanargmax(rpa_f1_list, axis=0)
    rpa_score_max = score_list[rpa_max_index]
    rpa_nu_max = nu_list[rpa_max_index]
    print('Best RPA threshold: {:.3f}'.format(rpa_nu_max))

    pa_max_index = np.nanargmax(pa_f1_list, axis=0)
    pa_score_max = score_list[pa_max_index]
    pa_nu_max = nu_list[pa_max_index]
    print('Best PA threshold:', pa_nu_max)

    pw_max_index = np.nanargmax(pw_f1_list, axis=0)
    pw_score_max = score_list[pw_max_index]

    predict = np.int64(normal_scores > rpa_nu_max)
    end_time = time.time()
    print("Execution time: {:.5f} seconds".format(end_time - start_time))

    return affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict

def ad_floating_ESA_DAMP(target, scores, if_aff):
    start_time = time.time()

    normal_scores = z_score(scores)
    events_gt = convert_vector_to_events(target)
    target_ts = create_time_series(target, chunk_size=100000)

    nu_list = np.arange(-3, 3, 0.01)
    pa_f1_list, pw_f1_list, rpa_f1_list = [], [], []
    affiliation_f1_list, affiliation_list = [], []
    score_list = []

    def _calc_f1_for_nu(detect_nu):
        predict = np.int64(normal_scores > detect_nu)
        if if_aff != 0:
            events_pred = convert_vector_to_events(predict)
            # 动态调整 Trange
            start_time = min(event[0] for event in events_gt + events_pred)  # 解包元组访问 start
            end_time = max(event[1] for event in events_gt + events_pred)    # 解包元组访问 end
            Trange = (start_time, end_time)
            dic = pr_from_events(events_pred, events_gt, Trange)
            affiliation_f1 = 2 * (dic["precision"] * dic["recall"]) / (dic["precision"] + dic["recall"])
        else:
            dic = {"precision": 0, "recall": 0}
            affiliation_f1 = 0
        predict_ts = create_time_series(predict, chunk_size=100000)
        score = accumulate_tsad_score(ground_truth=target_ts, predict=predict_ts)
        rpa_f1 = score.f1(ScoreType.RevisedPointAdjusted)
        pa_f1 = score.f1(ScoreType.PointAdjusted)
        pw_f1 = score.f1(ScoreType.Pointwise)
        return dic, affiliation_f1, rpa_f1, pa_f1, pw_f1, score

    results = Parallel(n_jobs=4)(
        delayed(_calc_f1_for_nu)(detect_nu) for detect_nu in nu_list
    )

    for item in results:
        dic, aff_f1, rpa_f1, pa_f1, pw_f1, sc = item
        affiliation_f1_list.append(aff_f1)
        affiliation_list.append(dic)
        rpa_f1_list.append(rpa_f1)
        pa_f1_list.append(pa_f1)
        pw_f1_list.append(pw_f1)
        score_list.append(sc)

    affiliation_f1_list = np.nan_to_num(affiliation_f1_list)
    affiliation_max_index = np.nanargmax(affiliation_f1_list, axis=0)
    affiliation_max = affiliation_list[affiliation_max_index]
    affiliation_nu_max = nu_list[affiliation_max_index]
    print("Best affiliation threshold: {:.3f}".format(affiliation_nu_max))

    rpa_max_index = np.nanargmax(rpa_f1_list, axis=0)
    rpa_score_max = score_list[rpa_max_index]
    rpa_nu_max = nu_list[rpa_max_index]
    print('Best RPA threshold: {:.3f}'.format(rpa_nu_max))

    pa_max_index = np.nanargmax(pa_f1_list, axis=0)
    pa_score_max = score_list[pa_max_index]
    pa_nu_max = nu_list[pa_max_index]
    print('Best PA threshold:', pa_nu_max)

    pw_max_index = np.nanargmax(pw_f1_list, axis=0)
    pw_score_max = score_list[pw_max_index]

    predict = np.int64(normal_scores > rpa_nu_max)
    end_time = time.time()
    print("Execution time: {:.5f} seconds".format(end_time - start_time))

    return affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict

# Anomaly score z_score standardization
def z_score(scores):
    mean = np.mean(scores)
    std = np.std(scores)
    if std != 0:
        scores = (scores - mean)/std
    return scores

def create_time_series(target, chunk_size=100000):
    """
    将 target 分块处理并逐步合并为一个 TimeSeries 对象。
    
    :param target: 输入的时间序列数据（NumPy 数组或列表）。
    :param chunk_size: 每次处理的块大小，默认为 100,000。
    :return: 合并后的 TimeSeries 对象。
    """
    combined_ts = None  # 初始化为空
    total_length = len(target)
# 一次性构造完整时间索引
    full_index = pd.date_range(start="1970-01-01", periods=total_length, freq="min")  # 每分钟一个点
    for i in range(0, total_length, chunk_size):
        # 当前块数据和对应时间戳
        chunk = target[i:i + chunk_size]
        timestamps = full_index[i:i + len(chunk)]
        # 构造 DataFrame
        chunk_df = pd.DataFrame({"value": chunk}, index=timestamps)

        # 构造 TimeSeries
        chunk_ts = TimeSeries.from_pd(chunk_df)
        # 合并
        if combined_ts is None:
            combined_ts = chunk_ts
        else:
            # 注意：需要处理时间轴对齐，直接按 Merlion 方式合并
            combined_ts += chunk_ts  # 合并 TimeSeries 
    return combined_ts


def create_time_series_lazy(target, chunk_size=100000):
    """
    延迟加载 target 数据，仅在需要时生成 TimeSeries 对象。
    
    :param target: 输入的时间序列数据（NumPy 数组或列表）。
    :param chunk_size: 每次处理的块大小，默认为 100,000。
    :return: 一个生成器，逐块返回 TimeSeries 对象。
    """
    total_length = len(target)

    for i in range(0, total_length, chunk_size):
        # 分块处理 target
        chunk = target[i:i + chunk_size]
        chunk_df = pd.DataFrame(chunk, columns=["value"])
        chunk_df["timestamp"] = pd.date_range(start=i, periods=len(chunk), freq="1T")  # 假设每分钟一个点
        yield TimeSeries.from_pd(chunk_df.set_index("timestamp"))

def ad_floating_lazy(target, scores, if_aff):
    start_time = time.time()

    normal_scores = z_score(scores)
    events_gt = convert_vector_to_events(target)

    # 使用生成器逐块生成 target_ts
    target_ts_generator = create_time_series_lazy(target, chunk_size=100000)

    nu_list = np.arange(-3, 3, 0.1)  # 减少范围或增大步长
    pa_f1_list, pw_f1_list, rpa_f1_list = [], [], []
    affiliation_f1_list, affiliation_list = [], []
    score_list = []

    def _calc_f1_for_nu(detect_nu):
        predict = np.int64(normal_scores > detect_nu)
        if if_aff != 0:
            events_pred = convert_vector_to_events(predict)
            Trange = (0, len(predict))
            dic = pr_from_events(events_pred, events_gt, Trange)
            affiliation_f1 = 2 * (dic["precision"] * dic["recall"]) / (dic["precision"] + dic["recall"])
        else:
            dic = {"precision": 0, "recall": 0}
            affiliation_f1 = 0

        # 使用生成器逐块生成 predict_ts
        predict_ts_generator = create_time_series_lazy(predict, chunk_size=100000)

        # 累积评分
        score = None
        for target_ts_chunk, predict_ts_chunk in zip(target_ts_generator, predict_ts_generator):
            if score is None:
                score = accumulate_tsad_score(ground_truth=target_ts_chunk, predict=predict_ts_chunk)
            else:
                score += accumulate_tsad_score(ground_truth=target_ts_chunk, predict=predict_ts_chunk)

        rpa_f1 = score.f1(ScoreType.RevisedPointAdjusted)
        pa_f1 = score.f1(ScoreType.PointAdjusted)
        pw_f1 = score.f1(ScoreType.Pointwise)
        return dic, affiliation_f1, rpa_f1, pa_f1, pw_f1, score

    results = Parallel(n_jobs=4)(
        delayed(_calc_f1_for_nu)(detect_nu) for detect_nu in nu_list
    )

    for item in results:
        dic, aff_f1, rpa_f1, pa_f1, pw_f1, sc = item
        affiliation_f1_list.append(aff_f1)
        affiliation_list.append(dic)
        rpa_f1_list.append(rpa_f1)
        pa_f1_list.append(pa_f1)
        pw_f1_list.append(pw_f1)
        score_list.append(sc)

    affiliation_f1_list = np.nan_to_num(affiliation_f1_list)
    affiliation_max_index = np.nanargmax(affiliation_f1_list, axis=0)
    affiliation_max = affiliation_list[affiliation_max_index]
    affiliation_nu_max = nu_list[affiliation_max_index]
    print("Best affiliation threshold: {:.3f}".format(affiliation_nu_max))

    rpa_max_index = np.nanargmax(rpa_f1_list, axis=0)
    rpa_score_max = score_list[rpa_max_index]
    rpa_nu_max = nu_list[rpa_max_index]
    print('Best RPA threshold: {:.3f}'.format(rpa_nu_max))

    pa_max_index = np.nanargmax(pa_f1_list, axis=0)
    pa_score_max = score_list[pa_max_index]
    pa_nu_max = nu_list[pa_max_index]
    print('Best PA threshold:', pa_nu_max)

    pw_max_index = np.nanargmax(pw_f1_list, axis=0)
    pw_score_max = score_list[pw_max_index]

    predict = np.int64(normal_scores > rpa_nu_max)
    end_time = time.time()
    print("Execution time: {:.5f} seconds".format(end_time - start_time))

    return affiliation_max, rpa_score_max, pa_score_max, pw_score_max, predict

