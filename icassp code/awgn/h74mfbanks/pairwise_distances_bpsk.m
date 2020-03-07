load mfbank_BPSK.mat;

rmfbank = real(mfbank);
e_h74 = sum(rmfbank.^2,2); % energy in each codeword
d_h74 = pdist(rmfbank); % euclidean distances
disp('Hamming code min | mean | max Euclidean distances...')
[min(d_h74) mean(d_h74) max(d_h74)]
figure(1)
subplot(2,1,1)
histogram(d_h74,2.9:0.1:5.5);
title('Hamming (7,4)');
disp('Only three unique values for Hamming code pairwise distances...')
unique(d_h74) % unique distances

load old_hamming_apr2019/ae_mfbank_BPSK.mat;

e_ae = sum(S_encoded_syms_bpsk.^2,2);
d_ae = pdist(S_encoded_syms_bpsk); % euclidean distances
disp('Autoencoder code min | mean | max Euclidean distances...')
[min(d_ae) mean(d_ae) max(d_ae)]
subplot(2,1,2)
histogram(d_ae,2.9:0.1:5.5);
xlabel('pairwise Euclidean distance')
title('autoencoder (7,4)'); 