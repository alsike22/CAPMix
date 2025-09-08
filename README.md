# CAPMix: Robust Time Series Anomaly Detection Based on Abnormal Assumptions with DualSpace Mixup
This repository provides the implementation of the _CAPMix: Robust Time Series Anomaly Detection Based on Abnormal Assumptions with DualSpace Mixup method_, called _CAPMix_ bellow. 

The implementation uses the Merlion libraries.

## Installation
This code is based on `Python 3.8`, all requires are written in `requirements.txt`. Additionally, we should install `saleforce-merlion v1.1.1` and `ts_dataset` as Merlion suggested.

```
pip install salesforce-merlion==1.1.1
pip install -r requirements.txt
```
This repository already includes the merlion's data loading package `ts_datasets`.
Download the data and unzip it in `data/iops_competition`, `data/ucr`, `data/swat`, `data/wadi`, `data/esa`, respectively.
e.g. For AIOps, download `phase2.zip` and unzip the `data/iops_competition/phase2.zip` before running the program.
For ESA dataset, download Mission 1 and unzip it in `data/esa`, then run `processingESAM1.py`.

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

