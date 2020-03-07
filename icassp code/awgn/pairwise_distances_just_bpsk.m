% This code computes all of the energies and pairwise distances for
% (7,4) Hamming codes and autoencoders only for BPSK
% DRB May 13, 2019
% For Asilomar extended abstract
% =================================================
% USER PARAMETERS BELOW
% =================================================
n = 7;                         % (n,k) block code
k = 4;                         % (n,k) block code

% n = 15;
% k = 11;
% =================================================
N = 2^k;                        % number of codewords
d_hamming = zeros((N^2-N)/2,1); % pairwise distances
d_ae = zeros((N^2-N)/2,1); % pairwise_distances
d_ae_matrix = zeros(N,N);

% load('h74mfbanks/mfbank_BPSK_7_4.mat'); % loads mfbank and B8
% load('h74mfbanks/ae_mfbank_hamming_AWGN_bpsk_drb4.mat');

% load('h1511mfbanks/mfbank_BPSK_15_11.mat');
% load('h1511mfbanks/ae_mfbank_AWGN_20191019_2222_150_15_11_bpsk.mat')

% load ae_mfbank_AWGN_20191016_1422_15_11_bpsk_v2.mat;
% The Hamming Matlab files contain two variables:
% B8 - 16x4 array containing the bit patterns associated with each codeword (irrelevant here)
% mfbank - 16x7 array containing the modulated codewords

% The autoencoder matlab files contain two variables:
% S_encoded_syms - 16x7 array containing the modulated codewords
% bit_dict - same as B8 above
        
% ENERGIES
e_hamming = sum(abs(mfbank).^2,2); % energy in each codeword
e_ae = sum(abs(S_encoded_syms).^2,2); % energy in each codeword
    
% PAIRWISE DISTANCES
ii = 0;
for i1 = 1:N
    for i2 = i1+1:N
        ii = ii + 1;
        d_hamming(ii) = norm(mfbank(i1,:)-mfbank(i2,:)); % euclidean distance
        d_ae(ii) = norm(S_encoded_syms(i1,:)-S_encoded_syms(i2,:)); % euclidean distance
    end
end

ii = 0;
for i1 = 1:N
    for i2 = 1:N
        d_ae_matrix(i1,i2) = norm(S_encoded_syms(i1,:)-S_encoded_syms(i2,:));
    end
end

% Confusion matrix at EbN0 dB
ae_conf_matrix = confusionmat(data_int,data_hat_ae_softdec_int);
sdd_conf_matrix = confusionmat(data_int,data_hat_softdec_int);
ae_err_matrix = ae_conf_matrix - diag(diag(ae_conf_matrix)); % take out diagonals
sdd_err_matrix = sdd_conf_matrix - diag(diag(sdd_conf_matrix)); % take out diagonals

% Energy table
fprintf('Energy statistics\n');
fprintf('BPSK Hamming codeword energies : min %5.3f | mean %5.3f | max %5.3f \n',min(e_hamming),mean(e_hamming),max(e_hamming));
fprintf('BPSK AutoEnc codeword energies : min %5.3f | mean %5.3f | max %5.3f \n',min(e_ae),mean(e_ae),max(e_ae));
fprintf('\n');
fprintf('Pairwise distance statistics\n');fprintf('BPSK Hamming pairwise distances: min %5.3f | mean %5.3f | max %5.3f \n',min(d_hamming),mean(d_hamming),max(d_hamming));
fprintf('BPSK AutoEnc pairwise distances: min %5.3f | mean %5.3f | max %5.3f \n',min(d_ae),mean(d_ae),max(d_ae));

BLER_sum = sum(ae_conf_matrix,2); % this is the same for all codewords
BLER = sum(ae_err_matrix,2)./BLER_sum;
[BLER_sorted,idx_BLER_sorted] = sort(BLER);

HBLER_sum = sum(sdd_conf_matrix,2); % this is the same for all codewords
HBLER = sum(sdd_err_matrix,2)./HBLER_sum;
[HBLER_sorted,idx_HBLER_sorted] = sort(HBLER);

%%
% BLER curve
figure(1);
plot(1:N,BLER_sorted,'d',1:N,HBLER_sorted,'+',[1 N],[sdec_BLER sdec_BLER],[1 N],[sdec_ae_BLER sdec_ae_BLER],'Linewidth',2,'Markersize',10);
grid on;
xlabel('i');
ylabel('BLER')
legend('AE-BLER','HBLER','overall HBLER','overall AE BLER');
 
% CDF
figure(2)
ecdf(d_ae);
hold on
ecdf(d_hamming);
title('CDF of AE and Hamming Distances');

% histogram
figure(3)
subplot(2,1,1)
histogram(d_hamming,3.2:0.1:7.4);
title('Distances with Soft Decision Decoder');
subplot(2,1,2)
histogram(d_ae,3.2:0.1:7.4);
xlabel('pairwise Euclidean distance')
title('Autoencoder Distances'); 