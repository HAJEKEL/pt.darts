import os
import logging
import shutil
import torch
import torchvision.datasets as dset
import numpy as np
import preproc
import pickle
data_path = './data/'
with open(data_path+'mnist1d_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(data['x'].shape)

dset_cls = dset.MNIST
print(dset_cls.shape)
