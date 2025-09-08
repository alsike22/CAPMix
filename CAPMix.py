import torch

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
from dataloader import *
from models.CAPMix.trainer.trainer_hidden_mix import Trainer
from models.CAPMix.network.model_hidden_mix import base_Model
from models.reasonable_metric import reasonable_accumulator
from ts_datasets.ts_datasets.anomaly import NAB, IOpsCompetition, SMAP, SMD, UCR
from tqdm import tqdm
from merlion.evaluate.anomaly import TSADScoreAccumulator as ScoreAcc, ScoreType
from utils import print_object
import matplotlib.pyplot as plt


# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('-ed','--experiment_description', default=f'Exp-{start_time}', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--visualization', default=False, type=bool,
                    help='Visualize')
parser.add_argument('--seed', default=4, type=int,
                    help='seed value')
parser.add_argument('--selected_dataset', default='WADI', type=str,
                    help='Dataset of choice: IOpsCompetition, UCR, NAB, SWaT, WADI')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
parser.add_argument('--ab_cap', type=int, default=0)
parser.add_argument('--ab_lr', type=str, default=0)
parser.add_argument('--ab_mix', type=int, default=0)
parser.add_argument("--config_override", type=str, default=None, help="Override config, e.g. alpha=0.1")
parser.add_argument("--pr", type=float, default=1, help="pollution rate in IOps")
args = parser.parse_args()

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'CutAddPaste'
run_description = args.run_description
selected_dataset = args.selected_dataset
visualization = args.visualization

exec(f'from conf.capmix.{data_type}_Configs import Config as Configs')
configs = Configs()
weight_decay = configs.weight

SEED = args.seed

print(f'Dataset: {data_type}')
print(f'Method:  {method}')
print(f'Random Seed:  {SEED}')
str_conf = print_object(configs)

# Load datasets
if selected_dataset in ['SWaT', 'WADI', 'ESA']:
    model_num = 1
else:
    if selected_dataset == 'NAB':
        dt = NAB()
    elif selected_dataset == 'IOpsCompetition':
        dt = IOpsCompetition()
    elif selected_dataset == 'SMAP':
        dt = SMAP()
    elif selected_dataset == 'UCR':
        dt = UCR()
    else:
        dt = SMD()
    model_num = len(dt)

# Aggregate statistics from full dataset
all_test_rpa_score, all_test_pa_score, all_test_pw_score = [], [], []
all_anomaly_num, all_test_scores_reasonable = [], []
all_test_aff_score, all_test_aff_precision, all_test_aff_recall = [], [], []
detect_list = np.zeros(model_num)
fid_of_all_train, fid_of_all_test, fid_of_all_test_clean = [], [], []
for idx in tqdm(range(model_num)):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(SEED)

    if selected_dataset == 'SWaT':
        train_data, test_data, train_labels, test_labels = swat()
    elif selected_dataset == 'WADI':
        train_data, test_data, train_labels, test_labels = wadi()
    elif selected_dataset == 'ESA':
        train_data, test_data, train_labels, test_labels = esa()
    else:
        time_series, meta_data = dt[idx]
        train_data, test_data, train_labels, test_labels = other_datasets(time_series, meta_data)

    print('>' * 32, len(train_data))
    print('>' * 32, len(test_data))

    # Load Model
    model = base_Model(configs).to(device)
    print("Data loaded ...")
    train_dl, val_dl, test_dl, test_anomaly_window_num = data_generator(train_data, test_data, train_labels,
                                                                         test_labels, SEED, configs)
    # train_dl, val_dl, test_dl, test_anomaly_window_num, fid_train_this_turn, fid_test_this_turn, fid_test_clean_this_turn, num_gt_1, num_lt_1 = data_generator_Mah(train_data, test_data, train_labels,
    #                                                                      test_labels, SEED, configs)
    # fid_of_all_train.append(fid_train_this_turn)
    # fid_of_all_test.append(fid_test_this_turn)
    # fid_of_all_test_clean.append(fid_test_clean_this_turn)

    model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=weight_decay)

    # Trainer
    test_score_origin, test_aff, test_rpa_score, test_pa_score, test_pw_score, score_reasonable, predict = Trainer(model, model_optimizer, train_dl,
                                                                                 val_dl, test_dl, device, configs, idx)

    all_anomaly_num.append(test_anomaly_window_num)
    all_test_scores_reasonable.append(score_reasonable)
    all_test_aff_precision.append(test_aff["precision"])
    all_test_aff_recall.append(test_aff["recall"])
    all_test_aff_score.append(test_aff)
    all_test_rpa_score.append(test_rpa_score)
    all_test_pa_score.append(test_pa_score)
    all_test_pw_score.append(test_pw_score)

    if visualization:
        anomaly_score_df = pd.DataFrame(test_score_origin, columns=['Anomaly Score'])
        anomaly_score_path = os.path.join("results", f"{method}_{selected_dataset}_subeset_{idx}.csv")
        anomaly_score_df.to_csv(anomaly_score_path, index=False)

all_anomaly_num = np.array(all_anomaly_num)
sum_anomaly_num = np.sum(all_anomaly_num)
all_test_aff_precision = np.array(all_test_aff_precision)
all_test_aff_precision = all_test_aff_precision * all_anomaly_num / sum_anomaly_num
test_aff_precision = np.nansum(all_test_aff_precision)
all_test_aff_recall = np.array(all_test_aff_recall)
all_test_aff_recall = all_test_aff_recall * all_anomaly_num / sum_anomaly_num
test_aff_recall = np.nansum(all_test_aff_recall)
if test_aff_precision + test_aff_recall == 0:
    test_aff_f1 = 0
else:
    test_aff_f1 = 2 * (test_aff_precision * test_aff_recall) / (test_aff_precision + test_aff_recall)

total_test_rpa_score = sum(all_test_rpa_score, ScoreAcc())
total_test_pa_score = sum(all_test_pa_score, ScoreAcc())
total_test_pw_score = sum(all_test_pw_score, ScoreAcc())
total_test_scores_reasonable = sum(all_test_scores_reasonable, reasonable_accumulator())
ucr_accuracy = total_test_scores_reasonable.get_all_metrics()

print('>' * 32)
if configs.dataset == 'UCR':
    print("UCR metrics:\n",
          f"accuracy: {ucr_accuracy}\n")
print("affiliation metrics:\n",
      f"Precision: {test_aff_precision:.5f}\n",
      f"Recall:    {test_aff_recall:.5f}\n"
      f"F1 Scores: {test_aff_f1:.5f}\n"
      "Revised-point-adjusted metrics:\n",
      f"Precision: {total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted):.5f}\n",
      f"Recall:    {total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted):.5f}\n"
      f"F1 score:  {total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted):.5f}\n",
      "Point-adjusted metrics:\n",
      f"Precision: {total_test_pa_score.precision(ScoreType.PointAdjusted):.5f}\n",
      f"Recall:    {total_test_pa_score.recall(ScoreType.PointAdjusted):.5f}\n",
      f"F1 score:  {total_test_pa_score.f1(ScoreType.PointAdjusted):.5f}\n",
      "Point-wise metrics:\n",
      f"Precision: {total_test_pw_score.precision(ScoreType.Pointwise):.5f}\n",
      f"Recall:    {total_test_pw_score.recall(ScoreType.Pointwise):.5f}\n"
      f"F1 Scores:  {total_test_pw_score.f1(ScoreType.Pointwise):.5f}\n"
      "NAB Scores:\n",
      f"NAB Score (balanced):       {total_test_pa_score.nab_score():.5f}\n",
      f"NAB Score (high precision): {total_test_pa_score.nab_score(fp_weight=0.22):.5f}\n",
      f"NAB Score (high recall):    {total_test_pa_score.nab_score(fn_weight=2.0):.5f}\n",
      "seed:", SEED, "\n"
      "config setup:\n"
      )
print_object(configs)
train_time = datetime.now()-start_time
print(f"Training time is : {train_time}")

path = "./results"
if not os.path.exists(path):
    os.makedirs(path)

summary = os.path.join("results", f"{method}_{selected_dataset}_summary_{experiment_description}.csv")
if os.path.exists(summary):
    df = pd.read_csv(summary, index_col=0)
else:
    df = pd.DataFrame()
model_name = method + f"{df.shape[1]}"

df.loc["Hyper-parameter", model_name] = str_conf
df.loc["seed", model_name] = SEED
df.loc["Train Time", model_name] = train_time
df.loc["UCR Accuracy", model_name] = round(ucr_accuracy["accuracy"], 5)
df.loc["Affiliation Precision", model_name] = round(test_aff_precision, 5)
df.loc["Affiliation Recall", model_name] = round(test_aff_recall, 5)
df.loc["Affiliation F1", model_name] = round(test_aff_f1, 5)
df.loc["RPA Precision", model_name] = round(total_test_rpa_score.precision(ScoreType.RevisedPointAdjusted), 5)
df.loc["RPA Recall", model_name] = round(total_test_rpa_score.recall(ScoreType.RevisedPointAdjusted), 5)
df.loc["RPA F1", model_name] = round(total_test_rpa_score.f1(ScoreType.RevisedPointAdjusted), 5)
df.loc["PA Precision", model_name] = round(total_test_pa_score.precision(ScoreType.PointAdjusted), 5)
df.loc["PA Recall", model_name] = round(total_test_pa_score.recall(ScoreType.PointAdjusted), 5)
df.loc["PA F1", model_name] = round(total_test_pa_score.f1(ScoreType.PointAdjusted), 5)
df.loc["Point-wise Precision", model_name] = round(total_test_pw_score.precision(ScoreType.Pointwise), 5)
df.loc["Point-wise Recall", model_name] = round(total_test_pw_score.recall(ScoreType.Pointwise), 5)
df.loc["Point-wise F1", model_name] = round(total_test_pw_score.f1(ScoreType.Pointwise), 5)
df.loc["NAB Score (balanced)", model_name] = round(total_test_rpa_score.nab_score(), 5)
df.loc["NAB Score (high precision)", model_name] = round(total_test_rpa_score.nab_score(fp_weight=0.22), 5)
df.loc["NAB Score (high recall)", model_name] = round(total_test_rpa_score.nab_score(fn_weight=2.0), 5)
df.to_csv(summary, index=True)
