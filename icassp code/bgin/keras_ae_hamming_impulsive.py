# -*- coding: utf-8 -*-
# Import libraries
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, GaussianNoise, Lambda, Add, BatchNormalization, Dropout, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
import os, time
import hdf5storage as h5

# Configure GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.Session(config=tf.ConfigProto())
K.set_session(sess)

def create_one_hot_encoded_data(N, M):
    label = np.random.randint(M,size=N)
    data = []
    for i in label:
        temp = np.zeros(M)
        temp[i] = 1
        data.append(temp)
    data = np.array(data)
    return label, data

def bernoulli_gaussian(noise_std_1, noise_std_2, N, n_channel, p):
    x1 = noise_std_1*np.random.randn(N,n_channel)
    x2 = noise_std_2*np.random.randn(N,n_channel)
    q = np.random.rand(N,n_channel)
    mask_bad_channel = 1*(q < p)
    mask_good_channel = 1*(q >= p)
    noise = mask_good_channel*x1 + mask_bad_channel*x2

    return noise

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
    adam = Adam(lr=1e-2)
    autoencoder.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    print (autoencoder.summary())
    
    encoder = Model(input_signal, encoded2)
    encoded_input = Input(shape=(n_channel,))  
    deco = autoencoder.layers[-2](encoded_input)
    deco = autoencoder.layers[-1](deco)
    decoder = Model(encoded_input, deco)

    return autoencoder, encoder, decoder


modulation_scheme = 'bpsk'
timestamp = '_20191013_0604_'
n_channel = 7
k = 4
R = 4/7
M = 2**k

N_train = 10**7
N_val = 10**5
N_test = 10**7

EbN0_dB_1 = 3.0
EbN0_dB_2 = -7.0
EbN0_1 = 10**(EbN0_dB_1/10)
noise_std_1 = 1/np.sqrt(2*R*EbN0_1)
EbN0_2 = 10**(EbN0_dB_2/10)
noise_std_2 = 1/np.sqrt(2*R*EbN0_2)

train_label, train_data = create_one_hot_encoded_data(N_train, M)
val_label, val_data =  create_one_hot_encoded_data(N_val, M)
test_label, test_data =  create_one_hot_encoded_data(N_test, M)

#p_vec = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
#prob_string = ['0','0point1','0point2','0point3','0point4','0point5','0point6','0point7','0point8','0point9','1']

p_vec = np.array([0.7,0.8,0.9,1])
prob_string = ['0point7','0point8','0point9','1']


train_noise = np.zeros((N_train,n_channel,len(p_vec)))
val_noise = np.zeros((N_val,n_channel,len(p_vec)))
test_noise = np.zeros((N_test,n_channel,len(p_vec)))

# Intialize other parameters
bler = np.zeros((len(p_vec),1))
for i in range(0,len(p_vec)):
    
    # Generate impulsive noise with specified p for train, val and test data
    train_noise = bernoulli_gaussian(noise_std_1, noise_std_2, N_train, n_channel, p_vec[i])
    val_noise = bernoulli_gaussian(noise_std_1, noise_std_2, N_val,n_channel, p_vec[i])
    
    autoencoder, encoder, decoder = keras_autoencoder_hamming(M, n_channel) 

    # Train the autoencoder - Keras
    history = autoencoder.fit([train_data, train_noise], train_data, 
                              epochs=50, batch_size= 2000,
                              validation_data=([val_data, val_noise], val_data), verbose = 1)
    bit_dict = d2b(np.arange(2**k), k)
    S_encoded_syms = np.zeros((2**k, 7))
    input_dict = np.eye(2**k)
    encoded_dict = encoder.predict(input_dict)
    S_encoded_syms = np.matrix(encoded_dict)
    dict1 = {}
    dict1['S_encoded_syms'] = S_encoded_syms  
    dict1['bit_dict'] = bit_dict.astype(np.int8)
    savemat('models_7_4/ae_mfbank_hamming_impulsive_'+prob_string[i]+timestamp+modulation_scheme+'_EbNodB1_'+str(EbN0_dB_1)+'_EbNodB2_'+str(EbN0_dB_2)+'.mat',dict1)    
    print('Generated dictionaries and encoded symbols for prob_' + prob_string[i])
    autoencoder.save('models_7_4/autoencoder_'+prob_string[i]+timestamp+modulation_scheme+'_EbNodB1_'+str(EbN0_dB_1)+'_EbNodB2_'+str(EbN0_dB_2)+'.h5')


    test_noise = bernoulli_gaussian(noise_std_1, noise_std_2, N_test,n_channel, p_vec[i])    
    encoded_signal = encoder.predict(test_data)
    noisy_signal = encoded_signal + test_noise
    decoded_signal =  decoder.predict(noisy_signal)
    decoded_output = np.argmax(decoded_signal,axis=1)
    no_errors = (decoded_output != test_label)
    no_errors =  no_errors.astype(int).sum()
    bler[i] =  no_errors / N_test 
    print('p:',p_vec[i], 'BLER:',bler[i])
    
    adict = {}
    adict['ae_BLER'] = bler
    savemat('models_7_4/ae_hamming_AWGN_bler_results_'+prob_string[i]+timestamp+modulation_scheme+'_EbNodB1_'+str(EbN0_dB_1)+'_EbNodB2_'+str(EbN0_dB_2)+'.mat', adict)
    K.clear_session()
    del autoencoder,encoder,decoder
