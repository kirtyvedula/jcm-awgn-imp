__author__ = 'kirtyvedula'

import time
from math import sqrt
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from utils import generate_encoded_sym_dict, get_args
from datasets import prepare_data

'''
TO DO:
1. Fix errors in validation loop - works fine, but shows wrong loss and accuracy values
2. Figure out how to view Tensorboard while training - install Tensorflow to run Tensorboard?
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
EbNo_test = torch.arange(0, 11.5, 0.5)
test_BLER = torch.zeros((len(EbNo_test), 1))

# Set sizes
train_set_size = 10 ** 5
# val_set_size = 10 ** 5
test_set_size = 10 ** 5

# CUDA for PyTorch - Makes sure the program runs on GPU when available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Define loggers
log_writer_train = SummaryWriter('logs/train/')
# log_writer_val = SummaryWriter('logs/val/')

class AWGN_Autoencoder(nn.Module):
    def __init__(self, k, n_channel):
        self.k = k
        self.n_channel = n_channel

        super(AWGN_Autoencoder, self).__init__()
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

    def forward(self, x):
        x_transmitted = self.transmitter(x)
        x_normalized = self.energy_normalize(x_transmitted)
        x_noisy = self.awgn(x_normalized)  # Gaussian Noise
        x = self.receiver(x_noisy)
        # x = x.to(device)
        return x

def validate_model(net,valloader,batch_size):
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
    return val_BLER, val_accuracy

def train_model(net,loss_func,optimizer,trainloader, log_writer_train,epochs, loss_vec):
    # Train the model
    start = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        net.train()
        for step, (x, y) in enumerate(trainloader):  # gives batch data

            # Move batches to GPU
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()  # clear gradients for this training step

            # This helps us export the messages at each stage and view how they evolve on Tensorboard.
            # Alternatively, we can just say output = net(x) if we just want to compute the final output
            x_transmitted = net.transmitter(x)
            x_normalized = net.energy_normalize(x_transmitted)
            x_noisy = net.awgn(x_normalized, EbN0_dB_train)
            output = net.receiver(x_noisy)
            loss = loss_func(output, y)  # Apply cross entropy loss

            # Backward and optimize
            loss.backward()  # back propagation, compute gradients
            optimizer.step()  # apply gradients
            loss_vec.append(loss.item())  # Append to loss_vec

            pred_labels = torch.max(output, 1)[1].data.squeeze()
            accuracy = sum(pred_labels == y) / float(batch_size)

            # statistics
            running_loss += loss.item()
            running_corrects += accuracy

        train_epoch_loss = running_loss / step
        train_epoch_acc = running_corrects/ step

        # # Validate model
        # val_BLER, val_accuracy = validate_model(net,valloader, epoch)
        print('Epoch: ', epoch + 1, '| train loss: %.4f' % train_epoch_loss, '| train acc: %4f' % (train_epoch_acc*100),'%')
        # print('Epoch: ', epoch + 1, '| train loss: %.4f' % train_epoch_loss, '| train acc: %4f' % (train_epoch_acc*100),'%','| val loss: %.4f' % val_BLER, '| val acc: %4f' % val_accuracy)
        log_writer_train.add_scalar('Train/Loss', train_epoch_loss, epoch)
        log_writer_train.add_scalar('Train/Accuracy', train_epoch_acc, epoch)
        # log_writer_val.add_scalar('Val/Loss', val_BLER, epoch)
        # log_writer_val.add_scalar('Val/Accuracy', val_accuracy, epoch)

    time_elapsed = time.time() - start
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return net

def run():
    torch.backends.cudnn.benchmark = True  # Make sure torch is accessing CuDNN libraries
    args =  get_args() # Get arguments - go with default (Hamming (7,4) BPSK) if not provided

    # Setup the model and move it to GPU
    net = AWGN_Autoencoder(k, n_channel)
    net = net.to(device)

    # Prepare training data
    traindataset, trainloader, train_labels = prepare_data(train_set_size, class_num, batch_size)
    valdataset, valloader, val_labels = prepare_data(val_set_size, class_num, val_set_size)  # Validation data

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # optimize all network parameters
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.01)   # Decay LR by a factor of 0.1 every 7 epochs
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    loss_vec = []

    trained_net = train_model(net,loss_func,optimizer,trainloader, log_writer_train,epochs, loss_vec)

    torch.save(trained_net.state_dict(), 'trained_net_74AE.ckpt')  # Save trained net
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
