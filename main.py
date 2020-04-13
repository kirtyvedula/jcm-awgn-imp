__author__ = 'kirtyvedula'

import torch

from awgn_train_test import awgn_train, awgn_test
from utils import prepare_data
from get_args import get_args

# User parameters
train_set_size = 10 ** 5
val_set_size = 10 ** 5
test_set_size = 10 ** 5

# CUDA for PyTorch - Makes sure the program runs on GPU when available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def run():
    torch.backends.cudnn.benchmark = True  # Make sure torch is accessing CuDNN libraries
    args = get_args()  # Get arguments - go with default (Hamming (7,4) BPSK) if not provided

    R = args.k / args.n_channel
    class_num = 2 ** args.k  # (n=7,k=4)  m=16

    # Prepare data
    traindataset, trainloader, train_labels = prepare_data(train_set_size, class_num, args.batch_size)
    valdataset, valloader, val_labels = prepare_data(val_set_size, class_num, val_set_size)  # Validation data
    testdataset, testloader, test_labels = prepare_data(test_set_size, class_num, test_set_size)

    # Training
    trained_net = awgn_train(trainloader, valloader, val_set_size, device, args)

    # TESTING
    awgn_test(testloader, trained_net, device, args)


if __name__ == '__main__':
    run()
