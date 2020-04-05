__author__ = 'kirtyvedula'

import time
from math import sqrt
import torch
import torch.nn as nn
import numpy as np

# Energy Constraints
def energy_constraint(x, args):
    # Energy Constraint
    n = (x.norm(dim=-1)[:, None].view(-1, 1).expand_as(x))
    x = sqrt(args.n_channel) * (x / n)
    return x

def awgn(x, args,EbN0_dB, device):
    SNR = 10 ** (EbN0_dB / 10)
    R = args.k / args.n_channel
    noise = torch.randn(x.size(), device= device) / ((2 * R * SNR) ** 0.5)
    noise.to(device)
    x += noise
    return x

def bgin(x, EbN0_dB_1, EbN0_dB_2, R, prob, device):
    SNR1 = 10 ** (EbN0_dB_1 / 10)
    SNR2 = 10 ** (EbN0_dB_2 / 10)
    noise_std_1 = 1 / ((2 * R * SNR1) ** 0.5)
    noise_std_2 = 1 / ((2 * R * SNR2) ** 0.5)

    x1 = noise_std_1 * torch.randn(x.size(), device=device)
    x2 = noise_std_2 * torch.randn(x.size(), device=device)
    q = torch.rand(x.size())
    mask_bad_channel = (1 * (q < prob)).to(device)
    mask_good_channel = (1 * (q >= prob)).to(device)
    noise = mask_good_channel * x1 + mask_bad_channel * x2
    noise.to(device)
    x += noise
    return x

# def interference( x, noise_shape, amp, omega, phase, type):
#     interference = torch.zeros(np.shape(noise_shape))
#     indices = torch.transpose(np.tile(np.arange(np.size(noise_shape, 0)), (np.size(noise_shape, 1), 1)))
#     if type == 'sin':
#         for i in range(np.size(indices, 1)):
#             interference[:, i] = amp * np.sin(omega * indices[:, i] + phase[:, i])
#     elif type == 'bpsk':
#         random_seq = np.random.randint(low=0, high=1, size=np.shape(noise_shape))
#         constellation = amp * 2 * (random_seq - 0.5)  # bpsk
#
#         indices = np.transpose(np.tile(np.arange(np.size(noise_shape, 0)), (np.size(noise_shape, 1), 1)))
#         interference = constellation * (np.exp(1j * omega * indices) + phase)
#     else:
#         print('Type not specified.')
#     x += interference
#     return x
#
