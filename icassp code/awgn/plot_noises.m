t = (0:0.1:10)';
x = sawtooth(t);
% Apply white Gaussian noise and plot the results.

% y = x + randn(length(x),1);


% Impulsive noise parameters
EbN0_dB_1 = 3.0;

EbN0_dB_2 = -7.0;
EbN0_dB_test = EbN0_dB_1;
R = k/n;
EbN0_1 = 10.^(EbN0_dB_1/10);
noise_std_1 = 1/sqrt(2*R*EbN0_1);
EbN0_2 = 10.^(EbN0_dB_2/10);
noise_std_2 = 1/sqrt(2*R*EbN0_2);

impulsive_noise = bernoulli_gaussian(noise_std_1, noise_std_2, length(x), 0.5);
y = x + impulsive_noise; % add complex impulsive noise
plot(t,[x y])
legend('Original Signal','With BGIN(3dB,-7dB,pb=0.5)')