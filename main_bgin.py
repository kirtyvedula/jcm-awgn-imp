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
from channels import awgn, energy_constraint
import numpy as np

'''
Version compatibility:
This works with the following versions installed:
tensorboard       2.0.0              
tensorboardx      2.0                  
pytorch           1.3.1           
cuda              9.2
cudnn             7.0    
'''

# User parameters
EbN0_dB_train = 3.0
# Hyperparameters
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

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # optimize all network parameters
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.01)   # Decay LR by a factor of 0.1 every 7 epochs
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    loss_vec = []

    start = time.time()
    for epoch in range(epochs):
        train_epoch_loss, train_epoch_acc = train(trainloader, net, optimizer, loss_func, device, loss_vec, batch_size, EbN0_dB_train, args)
        val_loss,  val_accuracy = validate(net,valloader,loss_func, val_set_size, device, EbN0_dB_train, args)
        print('Epoch: ', epoch + 1, '| train loss: %.4f' % train_epoch_loss, '| train acc: %4f' % (train_epoch_acc*100),'%','| val loss: %.4f' % val_loss, '| val acc: %4f' % (val_accuracy*100),'%')
        log_writer_train.add_scalar('Train/Loss', train_epoch_loss, epoch)
        log_writer_train.add_scalar('Train/Accuracy', train_epoch_acc, epoch)
        log_writer_val.add_scalar('Val/Loss', val_loss, epoch)
        log_writer_val.add_scalar('Val/Accuracy', val_accuracy, epoch)

        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    time_elapsed = time.time() - start
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    torch.save(net.state_dict(), 'trained_net_74AE.ckpt')  # Save trained net
    generate_encoded_sym_dict(n_channel, k, net, device)  # Generate encoded symbols

    # TESTING
    testdataset, testloader, test_labels = prepare_data(test_set_size, class_num, test_set_size, args)
    test_BLER = torch.zeros((len(EbN0_test), 1))

    for p in range(len(EbN0_test)):
        test_BLER[p] = test(net, testloader, device, EbN0_test[p])
        print('Eb/N0:', EbN0_test[p].numpy(), '| test BLER: %.4f' % test_BLER[p])

if __name__ == '__main__':
    run()
