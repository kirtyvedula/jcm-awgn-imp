__author__ = 'kirtyvedula'

import argparse

def get_args():
    ################################
    # Setup Parameters and get args
    ################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-n_channel', type=float, default = 7)
    parser.add_argument('-k', type=float, default = 4)
    parser.add_argument('-bec_p',type=float, default=0.0, help ='only for bec channel')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('-modulation', choices = ['BPSK','QPSK','8PSK','16QAM','64QAM','256QAM'], default='BPSK')
    parser.add_argument('-coding', choices=['SingleParity_4_3','Hamming_7_4','Hamming_15_11','Polar_16_4', 'EGolay_24_12'],default='Hamming_7_4')
    # parser.add_argument('-dropout',type=float, default=0.0)
    # parser.add_argument('-EbN0dB_test_start', type=float, default=-0.0)
    # parser.add_argument('-EbN0dB_test_end', type=float, default=11.0)
    # parser.add_argument('-EbN0dB_points', type=int, default=23)
    # parser.add_argument('-batch_size', type=int, default=100)
    # parser.add_argument('-num_epoch', type=int, default=1)
    args = parser.parse_args()

    return args
