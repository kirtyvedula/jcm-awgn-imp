__author__ = 'kirtyvedula'

import time
from math import sqrt
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from tools import EarlyStopping
from utils import generate_encoded_sym_dict, get_args
from datasets import prepare_data
from trainer import train, validate

'''
TO DO:
4. Add args to function for including all parameters - in determine_n_k(args)
'''

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
k = 4
n_channel = 7
EbN0_dB_train = 3.0

R = k / n_channel
class_num = 2 ** k  # (n=7,k=4)  m=16

# Hyperparameters
epochs = 50  # train the training data 'epoch' times
batch_size = 128
learning_rate = 0.001  # learning rate

# Test parameters
EbNo_test = torch.arange(0, 11.5, 0.5)
test_BLER = torch.zeros((len(EbNo_test), 1))

patience = 10   # early stopping patience; how long to wait after last time validation loss improved.
early_stopping = EarlyStopping(patience=patience, verbose=True) # initialize the early_stopping object

# Set sizes
train_set_size = 10 ** 5
val_set_size = 10 ** 4
test_set_size = 10 ** 5

# CUDA for PyTorch - Makes sure the program runs on GPU when available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Define loggers
log_writer_train = SummaryWriter('logs/train/')
log_writer_val = SummaryWriter('logs/val/')

class FC_Autoencoder(nn.Module):
    def __init__(self, k, n_channel):
        self.k = k
        self.n_channel = n_channel

        super(FC_Autoencoder, self).__init__()
        self.transmitter = nn.Sequential(
            nn.Linear(in_features=2 ** self.k, out_features=2 ** self.k, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2 ** self.k, out_features=self.n_channel, bias=True))
        self.receiver = nn.Sequential(
            nn.Linear(in_features=self.n_channel, out_features=2 ** self.k, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2 ** self.k, out_features=2 ** self.k, bias=True), )

    def energy_normalize(self, x):
        # Energy Constraint
        n = (x.norm(dim=-1)[:, None].view(-1, 1).expand_as(x))
        x = sqrt(self.n_channel) * (x / n)
        return x

    def awgn(self, x, EbN0_dB):
        SNR = 10 ** (EbN0_dB / 10)
        R = self.k / self.n_channel
        noise = torch.randn(x.size(), device=device) / ((2 * R * SNR) ** 0.5)
        x += noise
        return x

    def bgin(self, x, EbN0_dB_1, EbN0_dB_2, prob):
        SNR1 = 10 ** (EbN0_dB_1 / 10)
        SNR2 = 10 ** (EbN0_dB_2 / 10)
        noise_std_1 = 1/((2 * R * SNR1) ** 0.5)
        noise_std_2 = 1/((2 * R * SNR2) ** 0.5)

        x1 = noise_std_1*torch.randn(x.size(), device=device)
        x2 = noise_std_2*torch.randn(x.size(), device=device)
        q = torch.rand(x.size())
        mask_bad_channel = (1*(q < prob)).to(device)
        mask_good_channel = (1*(q >= prob)).to(device)
        noise = mask_good_channel*x1 + mask_bad_channel*x2
        noise.to(device)
        x += noise
        return x

    def forward(self, x):
        x_transmitted = self.transmitter(x)
        x_normalized = self.energy_normalize(x_transmitted)
        x_noisy = self.awgn(x_normalized)  # Gaussian Noise
        x = self.receiver(x_noisy)
        # x = x.to(device)
        return x

def run():
    torch.backends.cudnn.benchmark = True  # Make sure torch is accessing CuDNN libraries
    args =  get_args() # Get arguments - go with default (Hamming (7,4) BPSK) if not provided

    # Setup the model and move it to GPU
    net = FC_Autoencoder(k, n_channel)
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
        train_epoch_loss, train_epoch_acc = train(trainloader, net, optimizer,exp_lr_scheduler, loss_func, device, loss_vec, batch_size, EbN0_dB_train)
        val_loss, val_BLER, val_accuracy = validate(net,valloader,loss_func, val_set_size, device, EbN0_dB_train)
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

    # %% TESTING
    testdataset, testloader, test_labels = prepare_data(test_set_size, class_num, test_set_size)

    for p in range(len(EbNo_test)):
        net.eval()
        with torch.no_grad():
            for test_data, test_labels in testloader:
                test_data = test_data.to(device)
                test_labels = test_labels.to(device)

                encoded_signal = net.transmitter(test_data)
                constrained_encoded_signal = net.energy_normalize(encoded_signal)
                noisy_signal = net.awgn(constrained_encoded_signal, EbNo_test[p])
                decoded_signal = net.receiver(noisy_signal)

                pred_labels = torch.max(decoded_signal, 1)[1].data.squeeze()
                test_BLER[p] = sum(pred_labels != test_labels) / float(test_labels.size(0))

        print('Eb/N0:', EbNo_test[p].numpy(), '| test BLER: %.4f' % test_BLER[p])

if __name__ == '__main__':
    run()
