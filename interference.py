# Add interference
omega = 2*np.pi*-20*10**(-6)
amp = EcNo_sqrt_train
train_phase = np.random.uniform(-np.pi, np.pi,np.shape(train_noise))
val_phase =  np.random.uniform(-np.pi, np.pi,np.shape(val_noise))

train_interferer = sin_interferer(train_noise, amp, omega, train_phase)
val_interferer = sin_interferer(val_noise, amp, omega, val_phase)

def generate_interference():


# def interferer(noise, amp, omega, phase):
#     random_seq = np.random.randint(low=0, high=1, size=np.shape(noise))
#     constellation = iamp * 2 * (random_seq - 0.5) # bpsk

#     reshaped_noise = np.arange(np.size(noise, 0))
#     noisy_seq = np.transpose(np.tile(reshaped_noise, (np.size(noise, 1), 1)))
#     interferer = noise + constellation * (np.exp(1j * omega * noisy_seq) + phase)

#     return interferer

def sin_interferer(noise, amp, omega, phase):
    interferer = np.zeros(np.shape(noise))
    reshaped_noise = np.arange(np.size(noise, 0))
    indices = np.transpose(np.tile(reshaped_noise, (np.size(noise, 1), 1)))
    for i in range(np.size(indices,1)):
        interferer[:,i] = amp*np.sin(omega*indices[:,i] + phase[:,i])
    return interferer
