# Multi-Scale-Network-with-Two-Stage-Loss-for-SAR-Target-Recognition-with-Small-Training-Set
# ReadMe

## Requirements

Python >=3.6

Pytorch>=1.5.1

TensorboardX, sklearn, pandas, numpy, matplotlib, torchsummary, tqdm.

## Implementation

1. Bulid files under root directory, including './SavedFiles', './logs', './model'
2. Download MSTAR dataset from https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=targets, used 17 degrees as training set, and 15 degrees as testing set. Put the data in to new file './MSTAR_DataSet/17DEG_Train/' and './MSTAR_DataSet/15DEG_Test/' respectively.
3. Run jupyter notebook file: TrainbyGpu.ipynb to train the model.

