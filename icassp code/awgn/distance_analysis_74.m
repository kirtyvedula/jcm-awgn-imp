clc
clear
load('results_74_1E7_blocks_11012019.mat')

%%
N = 2^k;                        % number of codewords
d_hamming = zeros((N^2-N)/2,1); % pairwise distances
d_ae = zeros((N^2-N)/2,1); % pairwise_distances
d_ae_matrix = zeros(N,N);

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

% Energy table
fprintf('Energy statistics\n');
fprintf('BPSK Hamming codeword energies : min %5.3f | mean %5.3f | max %5.3f \n',min(e_hamming),mean(e_hamming),max(e_hamming));
fprintf('BPSK AutoEnc codeword energies : min %5.3f | mean %5.3f | max %5.3f \n',min(e_ae),mean(e_ae),max(e_ae));
fprintf('\n');
fprintf('Pairwise distance statistics\n');
fprintf('BPSK Hamming pairwise distances: min %5.3f | mean %5.3f | max %5.3f \n',min(d_hamming),mean(d_hamming),max(d_hamming));
fprintf('BPSK AutoEnc pairwise distances: min %5.3f | mean %5.3f | max %5.3f \n',min(d_ae),mean(d_ae),max(d_ae));


%%
% Find the minimum distances

d_ae_sorted = sort(d_ae,'ascend');
d_hamming_sorted = sort(d_hamming, 'ascend');
% d_ae_sorted_1 = d_ae_sorted(1:687); % The distances that are less than 3.464 (the min Hamming pairwise distance)
d_ae_sorted_1 = d_ae_sorted; % 
% Find where those errors come from

row = zeros(length(d_ae_sorted_1),1);
col = row;
num_errs_ae = row;
num_errs_sdd = row;

for i = 1: length(d_ae_sorted_1)
    [row(i),col(i)] = find(tril(d_ae_matrix) == d_ae_sorted_1(i));
    num_errs_ae(i) = ae_err_matrix(row(i),col(i));
    num_errs_sdd(i) = sdd_err_matrix(row(i),col(i));
    fprintf('count %7.0f | actual codeword %4.0f | mapped codeword %4.0f | errors with AE %3.0f | errors with SDD %3.0f\n',i, row(i)-1,col(i)-1, num_errs_ae(i),num_errs_sdd(i));
end

row_bin = de2bi(row-1);
col_bin = de2bi(col-1);
for i = 1 :length(d_ae_sorted_1)
disp([num2str(row_bin(i,:)),'  |  ', num2str(col_bin(i,:))])
end

%%
min_index = row(1); % 9 for (7,4) Hamming, 958 for (15,11) Hamming
cumsum_ae = cumsum(sort(ae_err_matrix(min_index,:),'descend'));
cumsum_sdd = cumsum(sort(sdd_err_matrix(min_index,:),'descend'));

%%

figure(1);
ae_err_matrix_sorted = sort(sum(ae_err_matrix,2), 'ascend');
sdd_err_matrix_sorted = sort(sum(sdd_err_matrix,2),'ascend');
plot(ae_err_matrix_sorted,sdd_err_matrix_sorted ,'r.', 'LineWidth',2); 
hold on;
plot([0e4 4e5], [0e4 4e5],'g')
axis square
grid on
% plot(d_ae_sorted, d_hamming_sorted,'r.')
% plot(sort(num_errs_ae,'ascend'), sort(num_errs_sdd, 'ascend'), 'r.');
% hold on
% plot([0 max(num_errs_ae)], [0 max(num_errs_ae)], 'g')
% axis square
% grid on
% xlabel('(7,4,7) Autoencoder Pairwise Distances');
% ylabel('(7,4) Hamming+BPSK Pairwise Distances')

% plot(1:N, cumsum_ae,'-d',1:N, cumsum_sdd,'-s', 'LineWidth',2,'markersize',5);
% grid on
% xlabel('i');
% ylabel('Cumulative Sum')
% xlim([0 N])
% legend('Cumulative Sum AE for codeword 9','Cumulative Sum SDD  for codeword 9','Location','se');