__author__ = 'kirtyvedula'

from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from scipy.io import savemat
from utils import d2b

# Hyperparameters
k = 4
n_channel = 7
R = k / n_channel
EbN0_dB_train = 3.0
class_num = 2 ** k  # (n=7,k=4)  m=16
epochs = 200  # train the training data e times
batch_size = 64
learning_rate = 0.001  # learning rate

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class FCAE_Interference(nn.Module):
    def __init__(self, k, n_channel):
        self.k = k
        self.n_channel = n_channel

        super(FCAE_Interference, self).__init__()
        self.transmitter = nn.Sequential(
            nn.Linear(in_features=2 ** self.k, out_features=2 ** self.k, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2 ** self.k, out_features=self.n_channel, bias=True))
        self.receiver = nn.Sequential(
            nn.Linear(in_features=self.n_channel, out_features=2 ** self.k, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2 ** self.k, out_features=2 ** self.k, bias=True), )

    # @staticmethod
    def energy_normalize(self, x):
        # Energy Constraint
        n = (x.norm(dim=-1)[:, None].view(-1, 1).expand_as(x))
        x = sqrt(self.n_channel) * (x / n)
        return x

    # @staticmethod
    def awgn(self, x, EbN0_dB):
        SNR = 10 ** (EbN0_dB / 10)
        R = self.k / self.n_channel
        noise = torch.randn(x.size(), device=device) / ((2 * R * SNR) ** 0.5)
        x += noise
        return x

    # @staticmethod
    def sin_interferer(self, x, amp, omega, phase):
        # Used for testing systematically
        indices = torch.from_numpy(np.transpose(np.tile(np.arange(np.size(x, 0)), (np.size(x, 1), 1))))
        interferer = amp*np.sin(omega*indices + phase)
        x += interferer
        return x

    # @staticmethod
    def domain_randomizer_sin_interferer(self, x):
        # Used to train
        omega = 2*np.pi*20*10**(-6)
        amp = 1/np.sqrt(2*(4/7)*(10**(5.0/10)))
        phase_dr = torch.from_numpy(np.random.uniform(-np.pi, np.pi,np.shape(x)))
        indices = torch.from_numpy(np.transpose(np.tile(np.arange(np.size(x, 0)), (np.size(x, 1), 1))))
        interferer = amp*np.sin(omega*indices + phase_dr)
        interferer = interferer.to(device)
        x += interferer
        return x

    def forward(self, x):
        x_transmitted = self.transmitter(x)
        x_normalized = self.energy_normalize(x_transmitted)
        x_noisy = self.awgn(x_normalized) # Gaussian Noise
        x_interfered = self. domain_randomizer_sin_interferer(x_noisy)
        x = self.receiver(x_interfered)
        # x = x.to(device)
        return x

def run():
    torch.backends.cudnn.benchmark = True
    net = FCAE_Interference(k, n_channel)
    net = net.to(device)

    # Train data
    train_set_size = 10 ** 5
    train_labels = (torch.rand(train_set_size) * class_num).long()
    train_data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=train_labels)
    traindataset = Data.TensorDataset(train_data, train_labels)
    trainloader = Data.DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    loss_vec = []

    # TRAINING
    for epoch in range(epochs):
        for step, (x, y) in enumerate(trainloader):  # gives batch data, normalize x when iterate train_loader

            x = x.to(device)
            y = y.to(device)

            x_transmitted = net.transmitter(x)
            x_normalized = net.energy_normalize(x_transmitted)
            x_noisy = net.awgn(x_normalized, EbN0_dB_train)
            x_interfered = net.domain_randomizer_sin_interferer(x_noisy)
            output = net.receiver(x_interfered)
            loss = loss_func(output, y)  # cross entropy loss

            # Backward and optimize
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            loss_vec.append(loss.item())

            pred_labels = torch.max(output, 1)[1].data.squeeze()
            accuracy = sum(pred_labels == y) / float(batch_size)

            if step % (10 ** 4) == 0:
                print('Epoch: ', epoch + 1, '| train loss: %.4f' % loss.item(), '| train acc: %4f' % accuracy)

    torch.save(net, '74AE_Interference.ckpt')  # Save model checkpoint

    # %% TESTING
    test_set_size = 10 ** 5
    test_labels = (torch.rand(test_set_size) * class_num).long()
    test_data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=test_labels)
    testdataset = Data.TensorDataset(test_data, test_labels)
    testloader = Data.DataLoader(dataset=testdataset, batch_size=test_set_size, shuffle=True, num_workers=4)

# Test 1: With varying EbN0


# Test 2: With varying phase

    # # Initialize outputs
    # EbNo_test = torch.arange(0, 11.5, 0.5)
    # test_BLER = torch.zeros((len(EbNo_test), 1))
    #
    # for p in range(len(EbNo_test)):
    #     net.eval()
    #     with torch.no_grad():
    #         for test_data, test_labels in testloader:
    #             test_data = test_data.to(device)
    #             test_labels = test_labels.to(device)
    #
    #             encoded_signal = net.transmitter(test_data)
    #             constrained_encoded_signal = net.energy_normalize(encoded_signal)
    #             noisy_signal = net.awgn(constrained_encoded_signal, EbNo_test[p])
    #             decoded_signal = net.receiver(noisy_signal)
    #
    #             pred_labels = torch.max(decoded_signal, 1)[1].data.squeeze()
    #             test_BLER[p] = sum(pred_labels != test_labels) / float(test_labels.size(0))
    #
    #     print('Eb/N0:', EbNo_test[p].numpy(), '| test BLER: %.4f' % test_BLER[p])

if __name__ == '__main__':
    run()
