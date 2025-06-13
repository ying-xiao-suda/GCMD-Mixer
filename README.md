# GCMD-Mixer


## Fine-Grained Day-Ahead Traffic Flow Prediction on Road Networks at Scale

### Abstract

Forecasting the traffic flow ahead of days is essential for travelers to manage important trips in advance and for governments to organize special events. However, day-ahead prediction cannot utilize the same manner as short-term prediction to model spatiotemporal dependencies of traffic flow data. The external influences from weather and date are also nonnegligible. Meanwhile, fine-grained forecasting needs to predict the traffic flow of all road segments and time slots for a future day, which requires manipulating high-dimensional inputs and outputs. 

To address these challenges, this paper proposes a graph convolutional mode decomposition with Mixer (GCMD-Mixer) model. Specifically, we decompose the traffic flow data within a day into the summation of daily variation patterns and day-specific fluctuations. Considering the cyclicity of traffic flow, its daily variation patterns are extracted from the frequency domain with a graph convolution network modeling spatial dependencies. Then, we leverage the Mixer on the constructed external feature matrix to capture day-specific fluctuations of traffic flow. Validated on four real-world datasets, GCMD-Mixer outperforms state-of-the-art traffic flow prediction methods by 3%-23%. 

---

## Experimental Platform

Experiments are conducted on a server with the following specifications:
- CPU: Intel Xeon Gold 5118, 2.30 GHz, 6 cores
- RAM: 64 GB
- GPU: NVIDIA Tesla A100, 40 GB
- Operating System: Ubuntu 22.04
- Python Version: 3.8.18
- Deep Learning Framework: PyTorch 2.1.2

All referenced models and code are executed within this environment.

## Directory Structure

```
Config/                   # Configuration files for different datasets (YAML format)
    hb.yaml
    pems04.yaml
    pems08.yaml
    shme.yaml
DataProcess/              # Processed data results
    hb_process_result/
    pems04_process_result/
    pems08_process_result/
    shme_process_result/
Datasets/                 # Raw or preprocessed datasets
    SHME/
    hb/
    pems04/
    pems08/
Model/                    # Model code
    gcmd_mxier.py
save/                     # Directory for saving experiment results or model checkpoints

README.md                 # Project documentation
dataset_hb.py             # Data processing script for hb dataset
dataset_pems04.py         # Data processing script for pems04 dataset
dataset_pems08.py         # Data processing script for pems08 dataset
dataset_shme.py           # Data processing script for shme dataset
exp_hb.py                 # Experiment script for hb dataset
exp_pems04.py             # Experiment script for pems04 dataset
exp_pems08.py             # Experiment script for pems08 dataset
exp_shme.py               # Experiment script for shme dataset
get_adj.py                # Script for generating adjacency matrices
run.sh                    # Shell script to run the main pipeline or experiments
utils.py                  # Utility functions
```


## Run experiments

The experiment scripts are in the `config` folder.

* `run.sh` is used to run experiments on Linux. An example of usage is

	```bash
	conda activate gcmd
	bash run.sh
	```

 
