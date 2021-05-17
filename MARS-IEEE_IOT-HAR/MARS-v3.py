# multi-model

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pickle as pkl
import matplotlib
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import torch.nn.functional as F
import itertools
import torch.utils.data as dataf
from sklearn import utils as skutils
import pandas as pd

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import MARS_Code.utils as utils

import torch
from torch import nn
import argparse

from sliding_window import sliding_window

parser = argparse.ArgumentParser(description='MARS_HAR')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-3)')
parser.add_argument('--imu-num', type=int, default=3,
                    metavar='N', help='Imu number (default: 3)')
parser.add_argument('--window-length', type=int, default=60,
                    metavar='N', help='sliding window length (default: 60)')
parser.add_argument('--window-step', type=int, default=30,
                    metavar='N', help='window step (default: 30)')
args = parser.parse_args()

global IMU_NUM
IMU_NUM = args.imu_num
NB_SENSOR_CHANNELS = IMU_NUM * 12
SLIDING_WINDOW_LENGTH = args.window_length
SLIDING_WINDOW_STEP = args.window_step
BATCH_SIZE = args.batch_size

# Load Data
x_train, y_train, x_test, y_test = utils.load_dataset('/home/xspc/Downloads/Pose_dataset/xspc_DIP_dataset/DIP_8_2/DIP_3IMU_82.pkl')

print('size of yt_train is:', y_train.shape)
print("size of xt_train is", x_train.shape)


assert NB_SENSOR_CHANNELS == x_train.shape[1]
x_train, y_train = utils.opp_sliding_window(x_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
x_test, y_test = utils.opp_sliding_window(x_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)

# Data is reshaped
X_train = np.array(x_train)
X_test = np.array(x_test)

X_train = X_train.reshape(-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)  # for input to Conv1D
X_test = X_test.reshape(-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)  # for input to Conv1D

print(" ..after sliding and reshaping, train data: inputs {0}, labels {1}".format(X_train.shape, y_train.shape))
print(" ..after sliding and reshaping, test data : inputs {0}, labels {1}".format(X_test.shape, y_test.shape))

X_train, y_train = skutils.shuffle(X_train, y_train, random_state=42)
X_test, y_test = skutils.shuffle(X_test, y_test, random_state=42)

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

dataset_train = utils.Dataset(X_train, y_train)
dataset_test = utils.Dataset(X_test, y_test)
train_loader = dataf.DataLoader(dataset_train, batch_size=100, drop_last=True)
test_loader = dataf.DataLoader(dataset_test, batch_size=100, drop_last=True)

class HARModel2(nn.Module):
    def __init__(self, n_hidden=96, n_layers=1, n_filters=32, stride=2, stride1D = 1, BATCH_SIZE = 100,
                 n_classes=5, filter_size=4, fusion = 2):   # out_channels= 32,
        super(HARModel2, self).__init__()
        self.fusion = fusion
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.filter_size = filter_size
        self.stride = stride
        self.stride1D = stride1D
        self.in_channels = NB_SENSOR_CHANNELS
        self.BATCH_SIZE = BATCH_SIZE
        # self.out_channels = out_channels

        self.net1d0 = nn.Sequential(
            nn.Conv1d(self.in_channels, self.n_filters, kernel_size=self.filter_size, stride=stride1D),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.Conv1d(self.n_filters, 2 * self.n_filters, kernel_size=self.filter_size, stride=stride1D),
            nn.BatchNorm1d(2 * n_filters),
            nn.ReLU(),
            nn.Conv1d(2 * self.n_filters, 3 * self.n_filters, kernel_size=self.filter_size, stride=stride1D),
            nn.BatchNorm1d(3 * self.n_filters),
            nn.ReLU(),
            nn.Conv1d(3 * self.n_filters, 4 * self.n_filters, kernel_size=self.filter_size, stride=stride1D),
            nn.BatchNorm1d(4 * n_filters),
            nn.ReLU(),
        )

        self.net1d3 = nn.Sequential(
            nn.ConvTranspose1d(4 * self.n_filters, 3 * self.n_filters, kernel_size=self.filter_size,
                               stride=stride1D, output_padding=stride1D - 1),
            nn.BatchNorm1d(3 * self.n_filters),
            nn.ReLU(),
            nn.ConvTranspose1d(3 * self.n_filters, 2 * self.n_filters, kernel_size=self.filter_size,
                               stride=stride1D),
            nn.BatchNorm1d(2 * n_filters),
            nn.ReLU(),
            nn.ConvTranspose1d(2 * self.n_filters, 1 * self.n_filters, kernel_size=self.filter_size,
                               stride=stride1D),
            nn.BatchNorm1d(1 * self.n_filters),
            nn.ReLU(),
            nn.ConvTranspose1d(self.n_filters, self.in_channels, kernel_size=self.filter_size,
                               stride=stride1D, output_padding=stride1D - 1),
        )
        self.net2d0 = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=self.filter_size, stride=self.stride),  #
            nn.BatchNorm2d(n_filters),
            nn.ReLU(),
            nn.Conv2d(n_filters, 2 * n_filters, kernel_size=self.filter_size, stride=self.stride),
            nn.BatchNorm2d(2 * n_filters),
            nn.ReLU(),
            nn.Conv2d(2 * n_filters, 2 * n_filters, kernel_size=self.filter_size, stride=self.stride-1),
            nn.BatchNorm2d(2 * n_filters),
            nn.ReLU(),
            nn.Conv2d(2 * n_filters, 3 * n_filters, kernel_size=self.filter_size, stride=self.stride-1),
            nn.BatchNorm2d(3 * n_filters),
            nn.ReLU(),
            # nn.Flatten()
        )

        self.net2d3 = nn.Sequential(
            nn.ConvTranspose2d(3 * n_filters, 2 * n_filters, kernel_size=self.filter_size, stride=self.stride-1),
            nn.BatchNorm2d(2 * n_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * n_filters, 2 * n_filters, kernel_size=self.filter_size, stride=self.stride-1),
            nn.BatchNorm2d(2 * n_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * n_filters, n_filters, kernel_size=self.filter_size, stride=self.stride, output_padding=(1,1)),
            nn.BatchNorm2d(1 * n_filters),
            nn.ReLU(),
            nn.ConvTranspose2d(n_filters, 1, kernel_size=self.filter_size, stride=self.stride),
        )

        self.net1d1 = nn.Sequential(
            nn.Linear(4 * self.n_filters * 48, self.n_hidden),
            nn.ReLU(),
        )

        self.net1d2 = nn.Sequential(
            nn.Linear(self.n_hidden, 4 * self.n_filters * 48),
            nn.ReLU(),
        )

        self.net2d1 = nn.Sequential(
            nn.Linear(672, self.n_hidden),
            nn.ReLU(),
        )

        self.net2d2 = nn.Sequential(
            nn.Linear(self.n_hidden, 672),
            nn.ReLU(),
        )

        self.netC = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_classes),
            nn.ReLU(),
            nn.Linear(self.n_classes, self.n_classes),
            # nn.ReLU(),
        )

        self.net_feature_fusion = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden),
            # nn.BatchNorm1d(self.n_classes),
            nn.Sigmoid(),
        )
        self.fc = nn.Sequential(
            nn.Linear(6816, 1024),
            # nn.BatchNorm1d(self.n_classes),
            nn.ReLU(),
            nn.Linear(1024, self.n_hidden),
            # nn.BatchNorm1d(self.n_classes),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_classes),
            # nn.BatchNorm1d(self.n_classes),
            nn.Sigmoid(),
        )

    # def latent_feature_fusion(self, latent_feature):

    #    latent_feature1 = nn.Linear(latent_feature.shape[1], self.n_hidden)


    #    return latent_feature_fused  ### size equals to batch size * n_hidden

    def KL_Distance(self, f1, f2):

        criterion_KL = nn.KLDivLoss(reduce=True)
        log_probs1 = F.log_softmax(f1, 1)
        probs1 = F.softmax(f1, 1)
        log_probs2 = F.log_softmax(f2, 1)
        probs2 = F.softmax(f2, 1)
        Distance_estimate = (criterion_KL(log_probs1, probs2) + criterion_KL(log_probs2, probs1))/2
        return Distance_estimate

    def forward(self, x):

        ### 1D
        x1d = x.view(-1, NB_SENSOR_CHANNELS, SLIDING_WINDOW_LENGTH)
        l1_1d = self.net1d0(x1d)
        l1_1d = l1_1d.view(self.BATCH_SIZE, -1)
        x1f = self.net1d1(l1_1d)
        l3_1d = self.net1d2(x1f)
        l3_1d = l3_1d.view(x1f.size(0), 4 * self.n_filters, 48)
        recon_x_1d = self.net1d3(l3_1d)
        recon_x_1d = recon_x_1d.view(-1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)

        ### 2D
        x2d = x.view(-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS)
        l1_2d = self.net2d0(x2d)
        l1_2d = l1_2d.view(self.BATCH_SIZE, -1)

        x2f = self.net2d1(l1_2d)

        l3_2d = self.net2d2(x2f)
        l3_2d = l3_2d.view(x2f.size(0), 3 * self.n_filters, 7, 1)
        recon_x_2d = self.net2d3(l3_2d)

        latent_feature1, latent_feature2 = self.net_feature_fusion(x1f), self.net_feature_fusion(x2f)

        # if self.fusion == 2:
        reduced_dim_x = self.netC(x1f.mul(self.net_feature_fusion(x1f)) + x2f.mul(1 - self.net_feature_fusion(x2f)))
        SDKL = self.KL_Distance(latent_feature1, latent_feature2)
        # print('size of SDKL is:', SDKL.shape)
        # reduced_dim_x = self.netC(x1f + x2f)  # fusion method II

        return recon_x_1d, reduced_dim_x, recon_x_2d, SDKL

    def init_hidden(self):
        """ Initializes hidden state """
        # Create two new tensors with sizes n_layers x BATCH_SIZE x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (nn.init.zeros_(weight.new(self.n_layers, self.BATCH_SIZE, self.n_hidden)).cuda(),
                      nn.init.zeros_(weight.new(self.n_layers, self.BATCH_SIZE, self.n_hidden)).cuda())
        else:
            hidden = (weight.new(self.n_layers, self.BATCH_SIZE, self.n_hidden).xavier_normal_(),
                      weight.new(self.n_layers, self.BATCH_SIZE, self.n_hidden).xavier_normal_())

        return hidden

net = HARModel2()

# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('Training on GPU!')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small')

opt = torch.optim.Adam(net.parameters(), lr=args.lr)
criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()

if train_on_gpu:
    net.cuda()

best_acc = 0


def train(epoch):

    net.train()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.99, patience=4)
    print("Learning rate：", opt.defaults['lr'])

    train_losses = []
    train_acc = 0

    batch_j = 0

    for i, data in enumerate(train_loader):  # 一个batch、一个batch的往下走，对于train，先走完

        x, y = data
        inputs, targets = x, y
        if train_on_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()

        opt.zero_grad()

        recon_output_1d, fused_output0, recon_output_2d, SDKL = net(inputs)

        output01d = torch.squeeze(recon_output_1d)
        output02d = torch.squeeze(recon_output_2d)

        _, pred = torch.max(fused_output0, 1)

        loss = criterion1(fused_output0, targets.long()) + 0.05 * criterion2(inputs, output01d) + \
               0.05 * criterion2(inputs, output02d) + 0.01 * SDKL  # 0.001 *   0.5 *

        train_losses.append(loss.item())
        train_acc += (pred == targets.long()).sum().item()

        loss.backward()  # 向后传播
        opt.step()

    print("第%d个epoch的学习率：%f" % (epoch + 1, opt.param_groups[0]['lr']))
    scheduler.step(loss)
    train_involved = (len(y_train) // BATCH_SIZE) * BATCH_SIZE
    print("Epoch: {}/{}...".format(epoch + 1, args.epochs),
          "Train Loss: {:.6f}...".format(np.mean(train_losses)),
          "Train Acc: {:.6f}...".format(train_acc / train_involved), end=" ")


def test(epoch):

    net.eval()
    val_accuracy = 0
    val_losses = []

    global best_acc

    with torch.no_grad():
        for i, data in enumerate(test_loader):  # 在一个batch中，一个个的往下走；

            x, y = data
            inputs, targets = x, y

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            _, fused_output, _, _ = net(inputs)

            _, predicted = torch.max(fused_output, 1)  # decision level fusion

            val_loss = criterion1(fused_output, targets.long())

            val_losses.append(val_loss.item())

            val_accuracy += (predicted == targets.long()).sum().item()

        test_involved = (len(y_test) // BATCH_SIZE) * BATCH_SIZE

        print("Val Loss: {:.6f}...".format(np.mean(val_losses)),
              "Val Acc: {:.6f}...".format(val_accuracy / test_involved))

        if best_acc < val_accuracy / test_involved:
            best_acc = val_accuracy / test_involved
            print("best model find: {:.6f}...".format(best_acc))
            torch.save(net,
                       '/home/xspc/Downloads/IMUPose_Code/xspc_test/MARS_Code/MARS_v3_result.pkl')
        else:
            print("no best model,the best is : {:.6f}...".format(best_acc))


try:
    for epoch in range(args.epochs):
        train(epoch)
        test(epoch)
    print("------Best Result--------")
    print("Val Acc: {:.6f}...".format(best_acc))
except KeyboardInterrupt:
    print("------Best Result--------")
    print("Val Acc: {:.6f}...".format(best_acc))


