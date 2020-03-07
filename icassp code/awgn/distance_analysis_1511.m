clc
clear
load('results_15_11_1E5_blocks_10302019.mat')


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

parpool(4);
parfor i = 1: length(d_ae_sorted_1)
    [row(i),col(i)] = find(tril(d_ae_matrix) == d_ae_sorted_1(i),1);
    num_errs_ae(i) = ae_err_matrix(row(i),col(i));
    num_errs_sdd(i) = sdd_err_matrix(row(i),col(i));
     fprintf('count %7.0f | actual codeword %4.0f | mapped codeword %4.0f | errors with AE %3.0f | errors with SDD %3.0f\n',i, row(i)-1,col(i)-1, num_errs_ae(i),num_errs_sdd(i));
end


row_bin = de2bi(row);
col_bin = de2bi(col);
% for i = 1 :length(d_ae_sorted_1)
% disp([num2str(row_bin(i,:)),'  |  ', num2str(col_bin(i,:))])
% end

%%
min_index = row(1); % 9 for (7,4) Hamming, 958 for (15,11) Hamming
cumsum_ae = cumsum(sort(ae_err_matrix(min_index,:),'descend'));
cumsum_sdd = cumsum(sort(sdd_err_matrix(min_index,:),'descend'));
% plot(1:N, cumsum_ae,'-d',1:N, cumsum_sdd,'-s', 'LineWidth',2,'markersize',3);
% grid on
% xlabel('index');
% ylabel('Cumulative Sum')
% legend('Cumulative Sum AE for codeword 958','Cumulative Sum SDD  for codeword 958','Location','se');
% 
% %%
% plot(1:N, cumsum_ae,'-d',1:N, cumsum_sdd,'-s', 'LineWidth',2,'markersize',3);
% grid on
% xlim([0 20])
% xlabel('index');
% ylabel('Cumulative Sum')
% legend('Cumulative Sum AE for codeword 958','Cumulative Sum SDD  for codeword 958','Location','se');

%%
figure(1);
% ae_err_matrix_sorted = sort(sum(ae_err_matrix,2)./sum(ae_conf_matrix,2), 'ascend');
% sdd_err_matrix_sorted = sort(sum(sdd_err_matrix,2)./sum(sdd_conf_matrix,2),'ascend');
ae_err_matrix_sorted = sort(sum(ae_err_matrix,2), 'ascend');
sdd_err_matrix_sorted = sort(sum(sdd_err_matrix,2),'ascend');

ae_greater_than_sdd_x = ae_err_matrix_sorted(sum(ae_err_matrix,2) > sum(sdd_err_matrix,2));
ae_greater_than_sdd_y = sdd_err_matrix_sorted(sum(ae_err_matrix,2) < sum(sdd_err_matrix,2));

plot(ae_err_matrix_sorted,sdd_err_matrix_sorted ,'r.', 'LineWidth',4); 
% hold on;
% plot(ae_greater_than_sdd_x, ae_greater_than_sdd_y,'b+', 'LineWidth',4);
% plot([0 3e4], [0 3e4],'g')
% plot([18421 27246], [18421 27246],'g')
axis square
grid on
% plot(d_ae_sorted, d_hamming_sorted,'ro')
% hold on
% plot([3 8], [3 8], 'g')

xlabel('No. of errors with (15,11,15) Autoencoder');
ylabel('No. of errors with (15,11) Hamming+BPSK')
saveas(gcf, 'errors_1511.pdf')

%%
% figure(2);
% plot(sort(num_errs_ae,'ascend'), sort(num_errs_sdd, 'ascend'), 'r.');
% hold on
% plot([0 2e4], [0 2e4], 'g')
% axis square
% grid on