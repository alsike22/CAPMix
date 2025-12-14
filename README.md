# CAPMix: Robust Time Series Anomaly Detection via Abnormal Assumptions with DualSpace Mixup
This repository provides the implementation of the _CAPMix: Robust Time Series Anomaly Detection via Abnormal Assumptions with DualSpace Mixup method_, called _CAPMix_ bellow. 

The implementation uses the Merlion libraries.

## Installation
This code is based on `Python 3.8`, all requirements are written in `requirements.txt`. Additionally, we should install `saleforce-merlion v1.1.1` and `ts_dataset` as Merlion suggested.

```
pip install salesforce-merlion==1.1.1
pip install -r requirements.txt
```
This repository already includes the merlion's data loading package `ts_datasets`.
Download the data and unzip it in `data/iops_competition`, `data/ucr`, `data/swat`, `data/wadi`, `data/esa`, respectively.
e.g. For AIOps, download `phase2.zip` and unzip the `data/iops_competition/phase2.zip` before running the program.
For the ESA dataset, download Mission 1 and unzip it in `data/esa`, then run `processingESAM1.py`.

## Repository Structure

### `conf`
This directory contains experiment parameters for all models on each dataset.

### `models`
Source code of CAPMix.

### `result`
Directory where the experiment result is saved.

## Usage
```
# CAPMix
python CAPMix.py --selected_dataset UCR --device cuda --seed 2
python CAPMix.py --selected_dataset IOpsCompetition --device cuda --seed 2
```

## Baselines
Anomaly Transformer(AnoTrans, AOT), AOC, RandomScore(RAS), NCAD, LSTMED, OC_SVM, IF, SR, RRCF, SVDD, DAMP, TS_AD(TCC)
RoCA, MixMamba, MTSCAD, AnomalyBert

We reiterate that in addition to our method, the source code of other baselines is based on the GitHub source code 
provided by their papers. For reproducibility, we changed the source code of their models as little as possible. 
We are grateful for the work on these papers.

We consult the GitHub source code of the paper corresponding to the baseline and then reproduce it. 
For baselines that use the same datasets as ours, we use their own recommended hyperparameters. 
For different datasets, we use the same hyperparameter optimization method Grid Search as our model to find the optimal hyperparameters.

## Acknowledgements
Part of the code, especially the baseline code, is based on the following source code.

[Anomaly Transformer(AnoTrans)](https://github.com/thuml/Anomaly-Transformer)

[AOC](https://github.com/alsike22/AOC)

[Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch)

[TS-TCC](https://github.com/emadeldeen24/TS-TCC)

[DAMP](https://sites.google.com/view/discord-aware-matrix-profile/documentation) and 
[DAMP-python](https://github.com/sihohan/DAMP)

LSTM_ED, SR, and IF are reproduced based on [saleforce-merlion](https://github.com/salesforce/Merlion/tree/main/merlion/models/anomaly)

[RRCF](https://github.com/kLabUM/rrcf?tab=readme-ov-file)

[MixMamba](https://github.com/KhaledAlkilane89/MixMamba)

[MTSCAD](https://github.com/BuckarooBanzay/mtscad)

[NCAD](https://github.com/awslabs/gluon-ts/tree/master/src/gluonts/nursery/ncad)

[RoCA](https://github.com/ruiking04/RoCA)

[AnomalyBert](https://github.com/Jhryu30/AnomalyBERT)



