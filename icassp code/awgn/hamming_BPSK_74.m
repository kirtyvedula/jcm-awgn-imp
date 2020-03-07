% Hamming code simulator
% Hard and soft decoding
% BPSK ONLY
% Moved the matched filter bank generator to a separate file
% DRB May 13, 2019
% =================================================
% USER PARAMETERS BELOW
% =================================================
Nbits = 4E7; % total number of bits
EbN0_dB_test = 3;
% EbN0_dB_test = 0:11;
n = 7;
k = 4;

% n = 15;
% k = 11;
%Nq = 1E5; % RCU: number of realizations of q
%Nt = 1E6; % RCU: number of realizations of tau/eta
% ======================**********************************9-===========================
R = k/n;
Nencbits = Nbits*n/k;
Nblocks = Nbits/k;

% diagnostic variables
unc_Es = zeros(1,length(EbN0_dB_test));
unc_Eb = zeros(1,length(EbN0_dB_test)); % 10*log10(unc_Eb/2) should match EbN0_dB_test
cod_Es = zeros(1,length(EbN0_dB_test));
cod_Eb = zeros(1,length(EbN0_dB_test)); % 10*log10((n/k)*cod_Eb/2) should match EbN0_dB_test
ae_cod_Es = zeros(1,length(EbN0_dB_test));
ae_cod_Eb = zeros(1,length(EbN0_dB_test)); % 10*log10((n/k)*cod_Eb/2) should match EbN0_dB_test

%% =========================================
% THEORY
% ==========================================
unc_SER_theory = zeros(1,length(EbN0_dB_test)); % symbol error rate of uncoded messages
unc_BER_theory = zeros(1,length(EbN0_dB_test)); % bit error rate of uncoded messages
cod_SER_theory = zeros(1,length(EbN0_dB_test)); % symbol error rate of coded messages before decoder
cod_BER_theory = zeros(1,length(EbN0_dB_test)); % bit error rate of coded messages before decoder
hdec_BLER_theory = zeros(1,length(EbN0_dB_test)); % hard decisions block error rate
rcu_BLER_theory = zeros(1,length(EbN0_dB_test)); % random coding union bound
na_BLER_theory = zeros(1,length(EbN0_dB_test)); % random coding union bound
mc_BLER_theory = zeros(1,length(EbN0_dB_test)); % random coding union bound

tic
i1 = 0;
for EbN0_dB = EbN0_dB_test
    i1 = i1+1;
    
    EbN0 = 10^(EbN0_dB/10);
    
    % UNCODED
    unc_SER_theory(i1) = qfunc(sqrt(2*EbN0)); % exact
    unc_BER_theory(i1) = qfunc(sqrt(2*EbN0)); % exact (BER = SER)
    
    % CODED (all calculations below are for bits/symbols before decoder)
    cod_SER_theory(i1) = qfunc(sqrt(2*EbN0*k/n)); % exact
    cod_BER_theory(i1) = qfunc(sqrt(2*EbN0*k/n)); % exact (BER = SER)
    
    % BLER: two or more bit errors results in a block error (Proakis p. 454)
    p = cod_BER_theory(i1);
    for kk=2:n
        hdec_BLER_theory(i1) = hdec_BLER_theory(i1) + nchoosek(n,kk)*p^kk*(1-p)^(n-kk);
    end
    
    % RCU BLER https://arxiv.org/pdf/1511.04629.pdf
    %     Omega = 2*R*EbN0;               % SNR
    %     q = randn(n,Nq)*sqrt(Omega/n);  % Nq realizations of q vector
    %     q(1,:) = q(1,:) + Omega;        % shift mean of first element in each realization
    %     normq = vecnorm(q);             % compute all of the norms outside of the for loop
    %     rho = q(1,:)./normq;            % ratio
    %     %tau = trnd(n-1,1,Nt);           % Nt realizations of tau (not sure I trust Matlab's student t distribution)
    %     g = randn(n,Nt);                % appendix F - Nt realizations of g vectors
    %     tau = g(1,:).*sqrt((n-1)./(vecnorm(g).^2-g(1,:).^2)); % Appendix F - Nt realizations of tau
    %     eta = tau./sqrt(n-1+tau.^2);    % Nt realizations of eta
    %     gq = zeros(1,Nq);
    %     for i=1:Nq
    %         gq(i) = sum(eta >= rho(i))/Nt; % P( ||q|| eta >= q1 ) = P( eta >= rho )
    %     end
    %     tmp = [ones(1,Nq) ; 2^(n*R)*gq];
    %    rcu_BLER_theory(i1) = mean(min(tmp));
    rcu_BLER_theory(i1) = rcu(EbN0,n,R);
    
    % NORMAL APPROXIMATION
    %     V = Omega*(2+Omega)/(2*(1+Omega)^2);  % dispersion
    %     C = 0.5*log2(1+Omega);
    %     na_BLER_theory(i1) = qfunc(sqrt(n/V)*((C-R)/log2(exp(1))+log(n)/2/n));
    na_BLER_theory(i1) = normalapprox(EbN0,n,R);
    
    % META CONVERSE
    %fun1 = @(x,n,Omega) pfa(x,n,Omega);  % parameterized function
    %fun2 = @(x) -(1/n)*log2(fun1(x,n,Omega))-R; % function
    %lambda = fzero(fun2,0.5);
    %mc_BLER_theory(i1) = pmd(lambda,n,Omega);
    mc_BLER_theory(i1) = metaconverse(EbN0,n,R);
    
    fprintf('Theory calculations progress: EbN0 (dB) %4.1f, time %6.1f\n',EbN0_dB,toc);
    
end

% %% =========================================
% % UNCODED SIMULATION
% % ==========================================
% unc_SER = zeros(2^k,length(EbN0_dB_test));
% unc_BER = zeros(2^k,length(EbN0_dB_test));
% 
% 
% %% Data
% % data = randi([0 1],Nbits,1); % generate a vector with all N bits
% % data_int = bi2de(reshape(data,k,Nblocks)'); % need this for BLER
% 
% 
% % A = repmat(0:2^k-1, 1,Nblocks/2^k);
% % % [~, cols] = size(A);
% % % idx = randperm(cols);
% % data_int = A';
% % % data_int(idx,1) = A(1,:);
% % data1 = de2bi(data_int);
% % data = reshape(data1',Nbits,1);
% 
% for i4 = 1: 2^k
%     data_int = (i4-1)* ones(Nblocks,1);
%     data1 = de2bi(data_int,k);
%     data = reshape(data1',Nbits,1);
%     
%     
%     
%     Nsyms = Nbits;
%     
%     tic
%     i1 = 0;
%     for EbN0_dB = EbN0_dB_test
%         i1 = i1+1;
%         
%         EbN0 = 10^(EbN0_dB/10);
%         x = data;
%         
%         % MODULATION (uncoded)
%         s = pskmod(x,2,0,'gray');
%         ss = s*sqrt(2*EbN0);  % scale (each symbol has energy k*Eb/n)
%         
%         % CHANNEL
%         y = ss + randn(length(s),1) + 1i*randn(length(s),1); % add complex noise
%         
%         % DIAGNOSTIC
%         unc_Es(i4,i1) = (ss'*ss)/length(ss);
%         unc_Eb(i4,i1) = unc_Es(i1);
%         
%         % DEMODULATION
%         zs = pskdemod(y,2,0,'gray'); % outputs are already binary
%         zb = zs;
%         
%         unc_SER(i4,i1) = symerr(x,zs)/Nsyms; % SER
%         unc_BER(i4,i1) = biterr(x,zs)/Nbits; % BER
%         
%         fprintf('Uncoded simulation progress: EbN0 (dB) %4.1f, codeword %4.0f, time %6.1f\n',EbN0_dB,(i4-1),toc);
%         
%     end
%     
% end
% 
% 
%% =========================================
% HAMMING CODED SIMULATION (use same data vector as for uncoded)
% ==========================================

cod_SER = zeros(2^k,length(EbN0_dB_test)); % before decoding
cod_BER = zeros(2^k,length(EbN0_dB_test)); % before decoding
hdec_BER = zeros(2^k,length(EbN0_dB_test)); % after decoding
hdec_BLER = zeros(2^k,length(EbN0_dB_test)); % after decoding
sdec_BER = zeros(2^k,length(EbN0_dB_test)); % after decoding
sdec_BLER = zeros(2^k,length(EbN0_dB_test)); % after decoding
sdec_ae_BER = zeros(2^k,length(EbN0_dB_test)); % after decoding
sdec_ae_BLER = zeros(2^k,length(EbN0_dB_test)); % after decoding

% NEED TO SAVE data_int,data_hat_ae_softdec_int, data_hat_softdec_int
data_hat_softdec_int_save = zeros(Nblocks, 1, 2^k);
data_hat_ae_softdec_int_save = zeros(Nblocks, 1, 2^k);
data_int_save = zeros(Nblocks, 1, 2^k);



% mfbank used for both coding/modulation and soft decision decoding

% % % 15-11
% load('h1511mfbanks/mfbank_BPSK_15_11.mat');
% load('h1511mfbanks/ae_mfbank_AWGN_20191019_2222_150_15_11_bpsk.mat')

% % 7-4
load('h74mfbanks/mfbank_BPSK_7_4.mat'); % loads mfbank and B8
load('h74mfbanks/ae_mfbank_hamming_AWGN_bpsk_drb4.mat'); % loads autoencoder S_encoded_syms_bpsk and bit_dict_bpsk

ae_mfbank = S_encoded_syms;
parpool(4)
parfor i4 = 1: 2^k
    
    tic
    data_int = (i4-1)* ones(Nblocks,1);
    data1 = de2bi(data_int,k);
    data = reshape(data1',Nbits,1);
%     clear data1
    % reshape into "blocks" with one block in each row and convert to integer indices
    data_sb_int = bi2de(reshape(data,k,Nblocks)');
    
    % Hamming joint coding and modulation (unit average energy per symbol here)
    s = zeros(n*Nblocks,1); % coded/modulated signal
    for i2 = 1:Nblocks
        index = data_sb_int(i2)+1;
        idx1 = (i2-1)*n+1;
        idx2 = i2*n;
        s(idx1:idx2) = mfbank(index,:);
    end
    
    % Autoencoder joint coding and modulation (unit average energy per symbol here)
    ae_s = zeros(n*Nblocks,1); % coded/modulated signal
    for i3 = 1:Nblocks
        index = data_sb_int(i3)+1;
        ae_s((i3-1)*n+1:i3*n) = ae_mfbank(index,:);
    end
    
    EbN0_dB = EbN0_dB_test;
    
    EbN0 = 10^(EbN0_dB/10);
    
    % now scale to proper EbN0 (each symbol has energy Eb*k/n)
    ss = s*sqrt(2*EbN0*k/n);  % Hamming
    ae_ss = ae_s*sqrt(2*EbN0*k/n);  % autoencoder
    
    % CHANNEL
    y = ss + randn(length(ss),1) + 1i*randn(length(ss),1); % add complex noise
    ae_y = ae_ss + randn(length(ae_ss),1) + 1i*randn(length(ae_ss),1); % add complex noise
    %y = ss;  % uncomment for debug
    %ae_y = ae_ss; % uncomment for debug
    
    % DIAGNOSTIC
    cod_Es(i4,i1) = (ss'*ss)/length(ss);
    cod_Eb(i4,i1) = cod_Es(i4,i1);
    ae_cod_Es(i4,i1) = (ae_ss'*ae_ss)/length(ae_ss);
    ae_cod_Eb(i4,i1) = ae_cod_Es(i4,i1);
    
    % HARD DECISION DEMODULATION (ONLY FOR HAMMING CODE)
    z = y/sqrt(2*EbN0*k/n); % scale
    ae_z = ae_y/sqrt(2*EbN0*k/n); % scale
    zs = pskdemod(z,2,0,'gray'); % BPSK 0,1
    
    % now run the Hamming decoder on the hard decisions
    zb = de2bi(zs);
    tmp = reshape(zb',n,Nblocks)'; % Nblocks by n
    data_hat = decode(tmp,n,k,'hamming/binary');  % Nblocks by k
    data_hat_int = bi2de(data_hat); % Nblocks by 1 (integers from 0 to 15 for each block)
    
    hdec_BER(i4,i1) = biterr(data_int,data_hat_int)/Nbits; % BER after decoder
    hdec_BLER(i4,i1) = symerr(data_int,data_hat_int)/Nblocks;  % BLER after hard decision decoder
    
    % SOFT DEMODULATION/DECODING
    % use z here (scaled channel output)
    
    data_hat_softdec_tmp = zeros(Nblocks,k);  % hamming
    data_hat_ae_softdec_tmp = zeros(Nblocks,k); % autoencoder
    oo = ones(1,2^k); % row of ones
    
    n1 = 1;
    n2 = n;  % always blocks of n symbols
    i5 = 0;
    while n2<=length(z) % z and ae_z should be the same length
        
        i5 = i5 + 1;
        
        % hamming
        t = z(n1:n2); % needs to be a column (n by 1)
        d = vecnorm(mfbank.' - t*oo);
        [~,imax] = min(d); % imax is the index of the closest codeword in the mfbank
        data_hat_softdec_tmp(i5,:) = double(B8(imax,:)); % k bits (B8 is int8 format)
        
        % autoencoder
        t = ae_z(n1:n2); % needs to be a column (n by 1)
        d = vecnorm(ae_mfbank.' - t*oo);
        [~,imax] = min(d); % imax is the index of the closest codeword in the mfbank
        data_hat_ae_softdec_tmp(i5,:) = double(B8(imax,:)); % k bits (B8 is int8 format)
        
        n1 = n1 + n;
        n2 = n2 + n;
        
    end
    
    % hamming
    data_hat_softdec = reshape(data_hat_softdec_tmp',k,Nblocks)'; %Nblocks by k
    data_hat_softdec_int = bi2de(data_hat_softdec); % Nblocks by 1 (integers from 0 to 15 for each block)
    sdec_BER(i4,i1) = biterr(data_int,data_hat_softdec_int)/Nbits; % BER after decoder
    sdec_BLER(i4,i1) = symerr(data_int,data_hat_softdec_int)/Nblocks;  % BLER after hard decision decoder
    
    % autoencoder
    data_hat_ae_softdec = reshape(data_hat_ae_softdec_tmp',k,Nblocks)'; %Nblocks by k
    data_hat_ae_softdec_int = bi2de(data_hat_ae_softdec); % Nblocks by 1 (integers from 0 to 15 for each block)
    sdec_ae_BER(i4,i1) = biterr(data_int,data_hat_ae_softdec_int)/Nbits; % BER after decoder
    sdec_ae_BLER(i4,i1) = symerr(data_int,data_hat_ae_softdec_int)/Nblocks;  % BLER after hard decision decoder
    
    fprintf('Coded simulation progress  : EbN0 (dB) %4.1f, codeword %4.0f, SDD BLER %1.3f, AE BLER %1.3f, time %6.1f\n',EbN0_dB, i4-1, sdec_BLER(i4,i1), sdec_ae_BLER(i4,i1), toc);
    
    data_hat_softdec_int_save(:,:,i4) = data_hat_softdec_int;
    data_hat_ae_softdec_int_save(:,:,i4) = data_hat_ae_softdec_int;
    data_int_save(:,:,i4) = data_int;
end



data_hat_softdec_int_save = reshape(data_hat_softdec_int_save, Nblocks*2^k,1);
data_hat_ae_softdec_int_save = reshape(data_hat_ae_softdec_int_save, Nblocks*2^k,1);
data_int_save = reshape(data_int_save, Nblocks*2^k,1);

clear ae_s ae_ss ae_y ae_z s x y z zb zs
clear data_int data_hat_ae_softdec data_hat_ae_softdec_int
clear data_hat_softdec data_hat_ae_softdec_tmp data_hat_ae_softdec_int data_hat_softdec_tmp
clear data data1 data_hat data_hat_int data_hat_softdec_int data_sb_int tmp ss
clear cod_Eb cod_Es cod_SER 

data_int_save = single(data_int_save);
data_hat_ae_softdec_int_save = single(data_hat_ae_softdec_int_save);
data_hat_softdec_int_save = single(data_hat_softdec_int_save);
% Confusion matrix
ae_conf_matrix = confusionmat(data_int_save,data_hat_ae_softdec_int_save);
sdd_conf_matrix = confusionmat(data_int_save,data_hat_softdec_int_save);
ae_err_matrix = ae_conf_matrix - diag(diag(ae_conf_matrix)); % take out diagonals
sdd_err_matrix = sdd_conf_matrix - diag(diag(sdd_conf_matrix)); % take out diagonals

BLER_sum = sum(ae_conf_matrix,2); % this is the same for all codewords
BLER = sum(ae_err_matrix,2)./BLER_sum;
[BLER_sorted,idx_BLER_sorted] = sort(BLER);

HBLER_sum = sum(sdd_conf_matrix,2); % this is the same for all codewords
HBLER = sum(sdd_err_matrix,2)./HBLER_sum;
[HBLER_sorted,idx_HBLER_sorted] = sort(HBLER);

%% PLOTS
N = 2^k;
% BLER curve
figure(1)
plot(1:N,BLER,'b.',1:N,HBLER,'.r',[1 N],[mean(sdec_BLER) mean(sdec_BLER)],'r',[1 N],[mean(sdec_ae_BLER) mean(sdec_ae_BLER)],'b','Linewidth',2,'Markersize',10);
grid on;
xlabel('i');
ylabel('BLER')
xlim([0 N])
legend('AE-BLER','HBLER','overall HBLER','overall AE BLER','Location','se');
% saveas(gcf,strcat('figs/H',num2str(n),num2str(k),'_BLER_curve'),'eps');
figure(2)
plot(1:N,BLER_sorted,'b.',1:N,HBLER_sorted,'.r',[1 N],[mean(sdec_BLER) mean(sdec_BLER)],'r',[1 N],[mean(sdec_ae_BLER) mean(sdec_ae_BLER)],'b','Linewidth',2,'Markersize',10);
grid on;
xlabel('i');
ylabel('BLER')
xlim([0 N])
legend('AE-BLER','HBLER','overall HBLER','overall AE BLER','Location','se');
% saveas(gcf,strcat('figs/H',num2str(n),num2str(k),'_BLER_curve_1'),'eps');

%%
figure;
semilogy(EbN0_dB_test,hdec_BLER,'-b^','LineWidth',2);
hold on
ax = gca;
ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,hdec_BLER_theory,'--b','Linewidth',2);
ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,sdec_BLER,'-k^','LineWidth',2);
ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,sdec_ae_BLER,':rd','Linewidth',2);
%ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,rcu_BLER_theory,'--m','Linewidth',2);
%ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,na_BLER_theory,'--c','Linewidth',2);
%ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,mc_BLER_theory,'--g','Linewidth',2);
legend('(7,4) Hamming HDD (sim)', '(7,4) Hamming HDD (theory)', ...
       '(7,4) Hamming SDD (sim)','(7,4) Autoencoder (sim)',...
       'RCU Bound (theory)','Normal Approximation (theory)','Metaconverse Bound (theory)','Location','southwest');

% legend('(15,11) Hamming HDD (sim)', '(15,11) Hamming HDD (theory)', ...
%        '(15,11) Hamming SDD (sim)','(15,11) Autoencoder (sim)',...
%        'RCU Bound (theory)','Normal Approximation (theory)','Metaconverse Bound (theory)','Location','southwest');
hold off
xlabel('Eb/N0 dB');
ylabel('Block Error Rate');
grid on
axis([min(EbN0_dB_test) max(EbN0_dB_test) 1E-7 1])
save('AutoEncoderResults_15_11_20191021_150_epochs_11E6.mat')
%%
% =================================================
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

% CDF
figure(3)
ecdf(d_ae);
hold on
ecdf(d_hamming);
title('CDF of AE and Hamming Distances');

% histogram
figure(4)
subplot(2,1,1)
histogram(d_hamming,3.2:0.1:7.4);
title('Hamming Distances)');
subplot(2,1,2)
histogram(d_ae,3.2:0.1:7.4);
xlabel('pairwise Euclidean distance')
title('Autoencoder Distances');
% saveas(gcf,strcat('figs/H',num2str(n),num2str(k),'_dist'),'eps')

save('results_74_1E7_blocks_11012019.mat')