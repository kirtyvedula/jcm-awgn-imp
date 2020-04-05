__author__ = 'kirtyvedula'

import time
from math import sqrt
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from models import FC_Autoencoder
from tools import EarlyStopping
from utils import generate_encoded_sym_dict, get_args
from datasets import prepare_data
from trainer import train, validate, test
from get_args import get_args
from awgn_train_test import awgn_train, awgn_test
from channels import awgn, energy_constraint
import numpy as np

# User parameters
EbN0_dB_train = 3.0
epochs = 100  # train the training data 'epoch' times
batch_size = 64
learning_rate = 0.001  # learning rate
EbN0_test = torch.arange(0, 11.5, 0.5)  # Test parameters

patience = 10   # early stopping patience; how long to wait after last time validation loss improved.
early_stopping = EarlyStopping(patience=patience, verbose=True) # initialize the early_stopping object

# Set sizes
train_set_size = 10 ** 5
val_set_size = 10 ** 5
test_set_size = 10 ** 5

# CUDA for PyTorch - Makes sure the program runs on GPU when available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Define loggers
log_writer_train = SummaryWriter('logs/train/')
log_writer_val = SummaryWriter('logs/val/')


def run():
    torch.backends.cudnn.benchmark = True  # Make sure torch is accessing CuDNN libraries
    args =  get_args() # Get arguments - go with default (Hamming (7,4) BPSK) if not provided
    R = args.k / args.n_channel
    class_num = 2 ** args.k  # (n=7,k=4)  m=16

    # Setup the model and move it to GPU
    net = FC_Autoencoder(args.k, args.n_channel)
    net = net.to(device)

    # Prepare data
    traindataset, trainloader, train_labels = prepare_data(train_set_size, class_num, batch_size)
    valdataset, valloader, val_labels = prepare_data(val_set_size, class_num, val_set_size)  # Validation data
    testdataset, testloader, test_labels = prepare_data(test_set_size, class_num, test_set_size)
    test_BLER = torch.zeros((len(EbN0_test), 1))

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # optimize all network parameters
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.01)   # Decay LR by a factor of 0.1 every 7 epochs
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    loss_vec = []

    # Training
    awgn_train(trainloader, valloader, val_set_size, epochs, net,
               optimizer, early_stopping, loss_func, device, loss_vec,
               batch_size, EbN0_dB_train, args, log_writer_train, log_writer_val)

    # TESTING
    awgn_test(testloader, net, device,  EbN0_test, test_BLER, args)
if __name__ == '__main__':
    run()
