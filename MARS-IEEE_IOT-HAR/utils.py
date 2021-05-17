import shutil

import numpy as np
from sklearn.metrics import accuracy_score

import torch

from sliding_window import sliding_window
import torch.utils.data as dataf
from MI_test.dim_losses import donsker_varadhan_loss, infonce_loss, fenchel_dual_loss

import numpy as np
import pickle as pkl
import pandas as pd


def load_dataset(filename):

    with open(filename, 'rb') as f:
        data = pkl.load(f, encoding='latin')
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


def opp_sliding_window(data_x, data_y, ws, ss):

    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])

    data_x = data_x.astype(np.float32)
    data_y = data_y.reshape(len(data_y)).astype(np.uint8)

    return data_x, data_y


class Dataset(dataf.Dataset): 

    def __init__(self, X, label):
        # TODO
        # 1. Initialize file path or list of file names.
        self.data_list = X
        self.data_label_list = label
        self.data_len = len(self.data_list)

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        return self.data_list[index], self.data_label_list[index]

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return self.data_len

