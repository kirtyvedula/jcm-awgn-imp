__author__ = 'kirtyvedula'

import time
from math import sqrt
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils import generate_encoded_sym_dict, get_args
from datasets import prepare_data
import numpy as np
from datetime import datetime

'''
Build a family of autoencoders for a given probability of a BGIN channel
'''

'''
TO DO:
1. Fix errors in validation loop - works fine, but shows wrong loss and accuracy values
2. Figure out how to view Tensorboard while training - install Tensorflow?
3. Add features like Early Stopping and Learning Rate Scheduler to train better
4. Add args to function for including all coding and modulation schemes - in determine_n_k(args)
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
R = k / n_channel
EbN0_dB_train = 3.0
class_num = 2 ** k  # (n=7,k=4)  m=16

# Hyperparameters
epochs = 30 # train the training data 'epoch' times
batch_size = 64
learning_rate = 0.001  # learning rate

# Test parameters
EbN0_dB_1 = 3.0
EbN0_dB_2 = -7.0

prob_vec = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
prob_string = ['0','0point1','0point2','0point3','0point4','0point5','0point6','0point7','0point8','0point9','1']

# Set sizes
train_set_size = 10 ** 5
# val_set_size = 10 ** 5
test_set_size = 10 ** 5

# CUDA for PyTorch - Makes sure the program runs on GPU when available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Define loggers
log_writer_train = SummaryWriter('logs/train/')
log_writer_val = SummaryWriter('logs/val/')

class BGIN_Autoencoder(nn.Module):
    def __init__(self, k, n_channel):
        self.k = k
        self.n_channel = n_channel

        super(BGIN_Autoencoder, self).__init__()
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


    def forward(self, x, prob):
        x_transmitted = self.transmitter(x)
        x_normalized = self.energy_normalize(x_transmitted)
        x_noisy = self.bgin(x_normalized)  # Gaussian Noise
        x = self.receiver(x_noisy)
        # x = x.to(device)
        return x

def validate_model(net,valloader,batch_size, log_writer_val):
    # Validation
    net.eval()
    with torch.no_grad():
        for val_data, val_labels in valloader:
            val_data = val_data.to(device)
            val_labels = val_labels.to(device)

            val_encoded_signal = net.transmitter(val_data)
            val_constrained_encoded_signal = net.energy_normalize(val_encoded_signal)
            val_noisy_signal = net.awgn(val_constrained_encoded_signal, EbN0_dB_train)
            val_decoded_signal = net.receiver(val_noisy_signal)

            val_pred_labels = torch.max(val_decoded_signal, 1)[1].data.squeeze()
            val_BLER = sum(val_pred_labels != val_labels) / float(val_labels.size(0))
            val_accuracy = sum(val_pred_labels == val_labels) / float(batch_size)

            log_writer_val.add_scalar('Val/Loss', val_BLER, epoch)
            log_writer_val.add_scalar('Val/Accuracy', val_accuracy, epoch)
    return val_BLER, val_accuracy

def train_model(net,loss_func,optimizer,trainloader, log_writer_train,epochs, loss_vec, prob):
    # Train the model
    start = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        net.train()
        for step, (x, y) in enumerate(trainloader):  # gives batch data
            x = x.to(device)     # Move batches to GPU
            y = y.to(device)

            optimizer.zero_grad()  # clear gradients for this training step

            # This helps us export the messages at each stage and view how they evolve on Tensorboard.
            # Alternatively, we can just say output = net(x) if we just want to compute the final output
            x_transmitted = net.transmitter(x)
            x_normalized = net.energy_normalize(x_transmitted)
            x_noisy = net.bgin(x_normalized, EbN0_dB_1, EbN0_dB_2, prob)
            output = net.receiver(x_noisy)
            loss = loss_func(output, y)  # Apply cross entropy loss

            # Backward and optimize
            loss.backward()  # back propagation, compute gradients
            optimizer.step()  # apply gradients
            loss_vec.append(loss.item())  # Append to loss_vec

            pred_labels = torch.max(output, 1)[1].data.squeeze()
            accuracy = sum(pred_labels == y) / float(batch_size)

            running_loss += loss.item() # statistics
            running_corrects += accuracy

        train_epoch_loss = running_loss / step
        train_epoch_acc = running_corrects/ step

        # # Validate model
        #
        print('Probability: ', prob, '| Epoch: ', epoch + 1, '| train loss: %.4f' % train_epoch_loss, '| train acc: %4f' % (train_epoch_acc*100),'%')

        log_writer_train.add_scalar('Train/Loss', train_epoch_loss, epoch)
        log_writer_train.add_scalar('Train/Accuracy', train_epoch_acc, epoch)


    time_elapsed = time.time() - start
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return net

def run():
    torch.backends.cudnn.benchmark = True  # Make sure torch is accessing CuDNN libraries
    args =  get_args() # Get arguments - go with default (Hamming (7,4) BPSK) if not provided


    now = datetime.now()    # current date and time
    timestamp = datetime.timestamp(now) # get timestamp from a datetime object

    # Prepare training data
    traindataset, trainloader, train_labels = prepare_data(train_set_size, class_num, batch_size)
    valdataset, valloader, val_labels = prepare_data(val_set_size, class_num, val_set_size)
    testdataset, testloader, test_labels = prepare_data(test_set_size, class_num, test_set_size)

    # Intialize other parameters
    bler = np.zeros((len(prob_vec),1))
    for i in range(0, len(prob_vec)):

        net = BGIN_Autoencoder(k, n_channel)    # Setup the model and move it to GPU
        net = net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # optimize all nn parameters
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.01)   # Decay LR by a factor of 0.1 every 7 epochs
        loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
        loss_vec = []

        trained_net = train_model(net,loss_func,optimizer,trainloader, log_writer_train,epochs, loss_vec, prob_vec[i])   # Training
        val_BLER, val_accuracy = validate_model(net,valloader, epoch)
        torch.save(trained_net.state_dict(), 'trained_nets/trained_net_74AE_'+str(timestamp)+'_prob_'+str(prob_string[i])+'.ckpt')  # Save trained net

            #
            # net.eval()
            # with torch.no_grad():
            #     for test_data, test_labels in testloader:
            #         test_data = test_data.to(device)
            #         test_labels = test_labels.to(device)
            #
            #         encoded_signal = net.transmitter(test_data)
            #         constrained_encoded_signal = net.energy_normalize(encoded_signal)
            #         noisy_signal = net.bgin(constrained_encoded_signal, EbN0_dB_1, EbN0_dB_2,prob)
            #         decoded_signal = net.receiver(noisy_signal)
            #
            #         pred_labels = torch.max(decoded_signal, 1)[1].data.squeeze()
            #         test_BLER[p] = sum(pred_labels != test_labels) / float(test_labels.size(0))
            #
            # print('Eb/N0:', EbNo_test[p].numpy(), '| test BLER: %.4f' % test_BLER[p])

if __name__ == '__main__':
    run()
