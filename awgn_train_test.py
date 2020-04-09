__author__ = 'kirtyvedula'

import time
from math import sqrt
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler

from models import FC_Autoencoder
from tools import EarlyStopping
from utils import generate_encoded_sym_dict
from datasets import prepare_data
from trainer import train, validate, test
from get_args import get_args

def awgn_train(trainloader, valloader, val_set_size, device, args):

    # Define loggers
    log_writer_train = SummaryWriter('logs/train/')
    log_writer_val = SummaryWriter('logs/val/')

    # Setup the model and move it to GPU
    net = FC_Autoencoder(args.k, args.n_channel)
    net = net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)  # optimize all network parameters
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.01)   # Decay LR by a factor of 0.1 every 7 epochs
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    patience = 10   # early stopping patience; how long to wait after last time validation loss improved.
    early_stopping = EarlyStopping(patience=patience, verbose=True) # initialize the early_stopping object
    loss_vec = []

    start = time.time()
    for epoch in range(args.epochs):
        train_epoch_loss, train_epoch_acc = train(trainloader, net, optimizer, loss_func, device, loss_vec, args.batch_size, args.EbN0_dB_train, args)
        val_loss,  val_accuracy = validate(net,valloader,loss_func, val_set_size, device, args.EbN0_dB_train, args)
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
    generate_encoded_sym_dict(args.n_channel, args.k, net, device)  # Generate encoded symbols

    return net

def awgn_test(testloader, net, device, args):
    EbN0_test = torch.arange(args.EbN0dB_test_start, args.EbN0dB_test_end, args.EbN0dB_precision)  # Test parameters
    test_BLER = torch.zeros((len(EbN0_test), 1))
    for p in range(len(EbN0_test)):
        test_BLER[p] = test(net, args, testloader, device, EbN0_test[p])
        print('Eb/N0:', EbN0_test[p].numpy(), '| test BLER: %.4f' % test_BLER[p])