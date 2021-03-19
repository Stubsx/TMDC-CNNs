# Multi-Scale-Network-with-Two-Stage-Loss-for-SAR-Target-Recognition-with-Small-Training-Set
# ReadMe
This code is the source code of the paper ***Multi-Scale Network with Two-Stage Loss for SAR Target Recognition with Small Training Set***.

## Requirements

Python >=3.6

Pytorch>=1.5.1

and the compatible version of TensorboardX, sklearn, pandas, numpy, matplotlib, torchsummary, tqdm and matplotlib.

## Implementation

1. Bulid new empty files under root directory, including './SavedFiles', './logs', and './model'
2. Download MSTAR dataset from https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=targets, used 17 degrees as training set, and 15 degrees as testing set. Put the data in to new file './MSTAR_DataSet/17DEG_Train/' and './MSTAR_DataSet/15DEG_Test/' respectively.
3. Run jupyter notebook file: TrainbyGpu.ipynb to train the model.

## Precautions

The results of our papers are the average results of 10 independently implementation of randomly selected training data. In order to facilitate others to get better results as soon as possible, we have added a new random data selection method and set it as the default. The data selected by this method can ensure a larger aspect angle range of the sample. For example, if you directly run TrainbyGPU.ipynb, you can get a recognition accuracy of 93.44% with 15 samples in each category which is much higher than the reported result in the paper(i.e., 88.61%).
