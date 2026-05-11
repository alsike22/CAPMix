# CAPMix: Robust KPI Anomaly Detection for AIOps in Noisy and Dynamic Environments
This repository provides the implementation of the _CAPMix: Robust KPI Anomaly Detection for AIOps in Noisy and Dynamic Environments_, called _CAPMix_ bellow. 

The implementation uses the Merlion libraries.

## Installation
This code is based on `Python 3.8`, all requires are written in `requirements.txt`. Additionally, we should install `saleforce-merlion v1.1.1` and `ts_dataset` as Merlion suggested.

```
pip install salesforce-merlion==1.1.1
pip install -r requirements.txt
```
This repository already includes the merlion's data loading package `ts_datasets`.
Download the data and unzip it in `data/iops_competition`, `data/asd`,`data/smd`,`data/exathlon`,`data/ucr`, `data/swat`, `data/wadi`, `data/esa`, `data/ack`, respectively.
e.g. For AIOps, download `phase2.zip` and unzip the `data/iops_competition/phase2.zip` before running the program.
For ESA dataset, download Mission 1 and unzip it in `data/esa`, then run `processingESAM1.py`.

## Dataset
We acknowledge the contributors of the dataset, including AIOps, UCR, SWaT, and WADI.
This repository already includes Merlion's data loading package `ts_datasets`.

### KPI datasets. 
1. AIOps Link: https://github.com/NetManAIOps/KPI-Anomaly-Detection .
Download and unzip the data in `data/iops_competition`. 
e.g. For AIOps, download `phase2.zip` and unzip the `data/iops_competition/phase2.zip` before running the program.
2. SMD and ASD Link: https://github.com/zhhlee/InterFusion/tree/main/data. 
The differences between the ASD and SMD datasets can be found at this link: https://github.com/zhhlee/InterFusion/blob/main/data/Dataset%20Description.pdf
3. Exathlon Link: https://github.com/exathlonbenchmark/exathlon

### Other time-series datasets. 
1. UCR Link: https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/ 
and https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip .
Download and unzip the data in `data/ucr`.
2. For SWaT and WADI, you need to apply by their official tutorial. Link: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ .
Because multiple versions of these two datasets exist, 
we used their newer versions: `SWaT.SWaT.A2_Dec2015, version 0` and `WADI.A2_19Nov2019`. Download and unzip the data in `data/swat` and `data/wadi` respectively. Then run the 
`swat_preprocessing()` and `wadi_preprocessing()` functions in `dataloader/data_preprocessing.py` for preprocessing.\
3. ESA Link: https://zenodo.org/records/12528696 and https://github.com/kplabs-pl/ESA-ADB

### AIClusterKPI (Ours). 
AIClusterKPI Link: https://zenodo.org/records/19926114

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

