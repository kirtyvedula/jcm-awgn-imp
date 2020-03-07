# Import libraries
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise, Lambda, Add, BatchNormalization, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import os
from sklearn.metrics.pairwise import euclidean_distances
from numpy.random import seed
from tensorflow import set_random_seed
# seed(5)
# set_random_seed(3.8)

# # Configure GPU
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# K.set_session(sess)

def d2b(d, n):
    d = np.array(d)
    d = np.reshape(d, (1, -1))
    power = np.flipud(2**np.arange(n))
    g = np.zeros((np.shape(d)[1], n))
    for i, num in enumerate(d[0]):
        g[i] = num * np.ones((1,n))
    b = np.floor((g%(2*power))/power)
    return np.fliplr(b)

def keras_autoencoder_hamming(M, n_channel):   
    input_signal = Input(shape=(M,))
    input_noise = Input(shape=(n_channel,))

    encoded = Dense(M, activation='relu')(input_signal)
    encoded1 = Dense(n_channel, activation='linear')(encoded)
    
    encoded2 = Lambda(lambda x: np.sqrt(n_channel) * K.l2_normalize(x, axis=1))(encoded1) #energy constraint
         
    #encoded2 = Lambda(lambda x: x / K.sqrt(K.mean(x**2)))(encoded1)   # average power constraint
    encoded_noise = Add()([encoded2, input_noise]) 

    decoded = Dense(M, activation='relu')(encoded_noise)
    decoded1 = Dense(M, activation='softmax')(decoded)

    autoencoder = Model(inputs=[input_signal,input_noise], outputs = decoded1)
    autoencoder.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print (autoencoder.summary())
    
    encoder = Model(input_signal, encoded2)
    encoded_input = Input(shape=(n_channel,))  
    deco = autoencoder.layers[-2](encoded_input)
    deco = autoencoder.layers[-1](deco)
    decoder = Model(encoded_input, deco)

    return autoencoder, encoder, decoder


def generate_dictionaries_and_encoded_syms(encoder,k):
    bit_dict = d2b(np.arange(2 ** k), k)
    S_encoded_syms = np.zeros((2**k, 7))
    input_dict = np.eye(2**k)
    encoded_dict = encoder.predict(input_dict)
    if k == 4:
        S_encoded_syms = np.matrix(encoded_dict)
    else:
        S_encoded_syms = np.matrix(encoded_dict[:,0:7]) + 1j*np.matrix(encoded_dict[:,7:14])
        S_encoded_syms = S_encoded_syms/np.sqrt(2)
    return S_encoded_syms, bit_dict 


def create_one_hot_encoded_data(N, M):
    label = np.random.randint(M,size=N)
    data = []
    for i in label:
        temp = np.zeros(M)
        temp[i] = 1
        data.append(temp)
    data = np.array(data)
    return label, data


modulation_scheme = 'bpsk'
k = 11
n_channel = 15
R = k/n_channel

M = 2**k

# Data parameters
N_train = 10**6
N_val = 10**5
N_test = 10**6

# Training parameters 
EbNo_dB_train = 3
EcNo_sqrt_train = 1/np.sqrt(2*R*(10**(EbNo_dB_train/10)))

# Initialize outputs
EbNo_test =np.arange(0, 11.5, 0.5)
bler = np.zeros((len(EbNo_test),1))

# Create data and noise for train and validation
train_label, train_data = create_one_hot_encoded_data(N_train, M)
val_label, val_data =  create_one_hot_encoded_data(N_val, M)
test_label, test_data =  create_one_hot_encoded_data(N_test, M)

train_noise = EcNo_sqrt_train * np.random.randn(N_train,n_channel)
val_noise = EcNo_sqrt_train * np.random.randn(N_val,n_channel)

# Setup the autoencoder - Keras
autoencoder, encoder, decoder = keras_autoencoder_hamming(M, n_channel) 

# Train the autoencoder - Keras
history = autoencoder.fit([train_data, train_noise], train_data, 
                          epochs=200, batch_size=1000,
                          validation_data=([val_data, val_noise], val_data), verbose = 1)

S_encoded_syms, bit_dict = generate_dictionaries_and_encoded_syms(encoder,k)
dict1 = {}
dict1['S_encoded_syms'] = S_encoded_syms  
dict1['bit_dict'] = bit_dict.astype(np.int8)
savemat('ae_mfbank_AWGN_'+str(n_channel)+'_'+str(k)+'_'+modulation_scheme+'.mat',dict1)    
print('Generated dictionaries and encoded symbols')
 ## Test the autoencoder
for p in range(len(EbNo_test)):
    EcNo_test_sqrt = 1/(2*R*(10**(EbNo_test[p]/20)))
    test_noise = EcNo_test_sqrt * np.random.randn(N_test, n_channel)
    encoded_signal = encoder.predict(test_data) 
    noisy_signal = encoded_signal + test_noise
    decoded_signal =  decoder.predict(noisy_signal)
    decoded_output = np.argmax(decoded_signal,axis=1)
    no_errors = (decoded_output != test_label)
    no_errors =  no_errors.astype(int).sum()
    bler[p] =  no_errors / N_test 
    print('Eb/N0:',EbNo_test[p], 'BLER:',bler[p])
 
adict = {}
adict['ae_BLER'] = bler
savemat('ae_hamming_AWGN_bler_results_'+str(n_channel)+'_'+str(k)+'_'+modulation_scheme+'.mat', adict)

