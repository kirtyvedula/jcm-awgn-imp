__author__ = 'kirtyvedula'

import torch
import numpy as np

def awgn(self, x, EbN0_dB, device, noise_shape):
    SNR = 10 ** (EbN0_dB / 10)
    R = self.k / self.n_channel
    noise = torch.randn(x.size(), device=device) / ((2 * R * SNR) ** 0.5)
    x += noise
    return x

def bgin(self, x, EbN0_dB_low, EbN0_dB_high, prob, device):
    R = self.k / self.n_channel
    SNR1 = 10 ** (EbN0_dB_low / 10)
    SNR2 = 10 ** (EbN0_dB_high / 10)
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

def generate_noise(noise_shape, args, test_sigma = 'default', snr_low = 0.0, snr_high = 0.0):
    fwd_noise = np.zeros(noise_shape)
        for batch_idx in range(noise_shape[0]):
            for code_idx in range(noise_shape[2]):

                good = True
                for time_idx in range(noise_shape[1]):
                    if good:
                        if test_sigma == 'default':
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_k[batch_idx,time_idx, code_idx]
                        else:
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_k
                        good = np.random.random()<p_gg
                    elif not good:
                        if test_sigma == 'default':
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_h[batch_idx,time_idx, code_idx]
                        else:
                            fwd_noise[batch_idx,time_idx, code_idx] = bsc_h
                        good = np.random.random()<p_bb
                    else:
                        print('bad!!! something happens')

        fwd_noise = torch.from_numpy(fwd_noise).type(torch.FloatTensor)* torch.randn(noise_shape, dtype=torch.float)

    elif args.channel == 'ge':
        #G-E discrete channel
        p_gg = 0.8         # stay in good state
        p_bb = 0.8
        bsc_k = 1.0        # accuracy on good state
        bsc_h = this_sigma# accuracy on good state

        fwd_noise = np.zeros(noise_shape)
        for batch_idx in range(noise_shape[0]):
            for code_idx in range(noise_shape[2]):

                good = True
                for time_idx in range(noise_shape[1]):
                    if good:
                        tmp = np.random.choice([0.0, 1.0], p=[1-bsc_k, bsc_k])
                        fwd_noise[batch_idx,time_idx, code_idx] = tmp
                        good = np.random.random()<p_gg
                    elif not good:
                        tmp = np.random.choice([0.0, 1.0], p=[ 1-bsc_h, bsc_h])
                        fwd_noise[batch_idx,time_idx, code_idx] = tmp
                        good = np.random.random()<p_bb
                    else:
                        print('bad!!! something happens')

        fwd_noise = torch.from_numpy(fwd_noise).type(torch.FloatTensor)

    else:
        # Unspecific channel, use AWGN channel.
        fwd_noise  = this_sigma * torch.randn(noise_shape, dtype=torch.float)

    return fwd_noise