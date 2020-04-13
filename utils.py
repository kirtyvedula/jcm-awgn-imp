import numpy as np
import torch
from scipy.io import savemat
import torch.utils.data as Data


def prepare_data(set_size, class_num, batch_size):
    '''
    trainloader, valloader and testloader are iterators used for batching, shuffling
    and loading the data in parallel using multiprocessing workers
    '''
    labels = (torch.rand(set_size) * class_num).long()
    data = torch.sparse.torch.eye(class_num).index_select(dim=0, index=labels)
    dataset = Data.TensorDataset(data, labels)
    loader = Data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataset, loader, labels

def d2b(d, n):
    d = np.array(d)
    d = np.reshape(d, (1, -1))
    power = np.flipud(2 ** np.arange(n))
    g = np.zeros((np.shape(d)[1], n))
    for i, num in enumerate(d[0]):
        g[i] = num * np.ones((1, n))
    b = np.floor((g % (2 * power)) / power)
    return np.fliplr(b)


def generate_encoded_sym_dict(n_channel, k, net, device):
    # Exporting Dictionaries
    bit_dict = d2b(torch.arange(2 ** k), k)
    input_dict = torch.eye(2 ** k).to(device)
    enc_output = net.transmitter(input_dict)
    S_encoded_syms = (enc_output.cpu()).detach().numpy()

    dict1 = {'S_encoded_syms': S_encoded_syms, 'bit_dict': bit_dict.astype(np.int8)}
    savemat('mfbanks/ae_mfbank_AWGN_BPSK_' + str(n_channel) + str(k) + '.mat', dict1)
    print('Generated dictionaries and encoded symbols')

# def get_plots():
#     # Plot 1 -
#     plt.plot(train_acc_store,'r-o')
#     plt.plot(test_acc_store,'b-o')
#     plt.xlabel('number of epochs')
#     plt.ylabel('accuracy')
#     plt.ylim(0.85,1)
#     plt.legend(('training','validation'),loc='upper left')
#     plt.title('train and test accuracy w.r.t epochs')
#     plt.show()
#
#     # Plot 2 -
#     plt.plot(train_loss_store,'r-o')
#     plt.plot(test_loss_store,'b-o')
#     plt.xlabel('number of epochs')
#     plt.ylabel('loss')
#     plt.legend(('training','validation'),loc='upper right')
#     plt.title('train and test loss w.r.t epochs')
#     plt.show()
