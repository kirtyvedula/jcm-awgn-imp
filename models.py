__author__ = 'kirtyvedula'

import torch.nn as nn


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

    def forward(self, x):
        # x_transmitted = self.transmitter(x)
        # x_normalized = self.energy_normalize(x_transmitted)
        # x_noisy = self.awgn(x_normalized)  # Gaussian Noise
        # x = self.receiver(x_noisy)
        # # x = x.to(device)

        # No need for this as we won't be using this anyway
        return x
