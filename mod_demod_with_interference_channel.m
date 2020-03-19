% Hamming code simulator
% Hard and soft decoding
% BPSK ONLY
% Moved the matched filter bank generator to a separate file
% DRB May 13, 2019
% =================================================
% USER PARAMETERS BELOW
% =================================================
Nbits = 1E6; % total number of bits
EbN0_dB_test = 0:0.5:11;
n = 7;
k = 4;
Nq = 1E4; % RCU: number of realizations of q
Nt = 1E4; % RCU: number of realizations of tau/eta
% =================================================
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
    Omega = 2*R*EbN0;               % SNR
    q = randn(n,Nq)*sqrt(Omega/n);  % Nq realizations of q vector
    q(1,:) = q(1,:) + Omega;        % shift mean of first element in each realization   
    normq = vecnorm(q);             % compute all of the norms outside of the for loop
    rho = q(1,:)./normq;            % ratio 
    %tau = trnd(n-1,1,Nt);           % Nt realizations of tau (not sure I trust Matlab's student t distribution)
    g = randn(n,Nt);                % appendix F - Nt realizations of g vectors
    tau = g(1,:).*sqrt((n-1)./(vecnorm(g).^2-g(1,:).^2)); % Appendix F - Nt realizations of tau
    eta = tau./sqrt(n-1+tau.^2);    % Nt realizations of eta
    gq = zeros(1,Nq);
    for i=1:Nq
        gq(i) = sum(eta >= rho(i))/Nt; % P( ||q|| eta >= q1 ) = P( eta >= rho 3)
    end
    tmp = [ones(1,Nq) ; 2^(n*R)*gq];
    rcu_BLER_theory(i1) = mean(min(tmp));
    
    % NORMAL APPROXIMATION
    V = Omega*(2+Omega)/(2*(1+Omega)^2);  % dispersion
    C = 0.5*log2(1+Omega);
    na_BLER_theory(i1) = qfunc(sqrt(n/V)*((C-R)/log2(exp(1))+log(n)/2/n));
    
    % META CONVERSE
    fun1 = @(x,n,Omega) pfa(x,n,Omega);  % parameterized function
    fun2 = @(x) -(1/n)*log2(fun1(x,n,Omega))-R; % function
    lambda = fzero(fun2,0.5);
    mc_BLER_theory(i1) = pmd(lambda,n,Omega);    
    
    fprintf('Theory calculations progress: EbN0 (dB) %4.1f, time %6.1f\n',EbN0_dB,toc);
    
end

%% =========================================
% UNCODED SIMULATION
% ==========================================
unc_SER = zeros(1,length(EbN0_dB_test));
unc_BER = zeros(1,length(EbN0_dB_test));

data = randi([0 1],Nbits,1); % generate a vector with all N bits
data_int = bi2de(reshape(data,k,Nblocks)'); % need this for BLER

Nsyms = Nbits;

tic    
i1 = 0;
for EbN0_dB = EbN0_dB_test
    i1 = i1+1;
    
    EbN0 = 10^(EbN0_dB/10);
    x = data;
    
    % MODULATION (uncoded)
    s = pskmod(x,2,0,'gray');
    ss = s*sqrt(2*EbN0);  % scale (each symbol has energy k*Eb/n)
        
    % CHANNEL
    y = ss + randn(length(s),1) + 1i*randn(length(s),1); % add complex noise
    
    % DIAGNOSTIC
    unc_Es(i1) = (ss'*ss)/length(ss);
    unc_Eb(i1) = unc_Es(i1);
    
    % DEMODULATION
    zs = pskdemod(y,2,0,'gray'); % outputs are already binary
    zb = zs;
    
    unc_SER(i1) = symerr(x,zs)/Nsyms; % SER
    unc_BER(i1) = biterr(x,zs)/Nbits; % BER
    
    fprintf('Uncoded simulation progress: EbN0 (dB) %4.1f, time %6.1f\n',EbN0_dB,toc);
    
end
    

%% =========================================
% HAMMING CODED SIMULATION (use same data vector as for uncoded)
% ==========================================
cod_SER = zeros(1,length(EbN0_dB_test)); % before decoding
cod_BER = zeros(1,length(EbN0_dB_test)); % before decoding
hdec_BER = zeros(1,length(EbN0_dB_test)); % after decoding
hdec_BLER = zeros(1,length(EbN0_dB_test)); % after decoding
sdec_BER = zeros(1,length(EbN0_dB_test)); % after decoding
sdec_BLER = zeros(1,length(EbN0_dB_test)); % after decoding
sdec_ae_BER = zeros(1,length(EbN0_dB_test)); % after decoding
sdec_ae_BLER = zeros(1,length(EbN0_dB_test)); % after decoding

tic

% mfbank used for both coding/modulation and soft decision decoding
load('mfbank_BPSK.mat'); % loads mfbank and B8
load('ae_mfbank_hamming_AWGN_bpsk_drb4.mat'); % loads autoencoder S_encoded_syms_bpsk and bit_dict_bpsk
ae_mfbank = S_encoded_syms;

% reshape into "blocks" with one block in each row and convert to integer indices
data_sb_int = bi2de(reshape(data,k,Nblocks)');

% Hamming joint coding and modulation (unit average energy per symbol here)
s = zeros(n*Nblocks,1); % coded/modulated signal
for i2 = 1:Nblocks
    index = data_sb_int(i2)+1;
    s((i2-1)*n+1:i2*n) = mfbank(index,:);
end

% Autoencoder joint coding and modulation (unit average energy per symbol here)
ae_s = zeros(n*Nblocks,1); % coded/modulated signal
for i2 = 1:Nblocks
    index = data_sb_int(i2)+1;
    ae_s((i2-1)*n+1:i2*n) = ae_mfbank(index,:);
end

i1 = 0;
for EbN0_dB = EbN0_dB_test
    i1 = i1+1;
    
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
    cod_Es(i1) = (ss'*ss)/length(ss);
    cod_Eb(i1) = cod_Es(i1);
    ae_cod_Es(i1) = (ae_ss'*ae_ss)/length(ae_ss);
    ae_cod_Eb(i1) = ae_cod_Es(i1);
    
    % HARD DECISION DEMODULATION (ONLY FOR HAMMING CODE)
    z = y/sqrt(2*EbN0*k/n); % scale
    ae_z = ae_y/sqrt(2*EbN0*k/n); % scale
    zs = pskdemod(z,2,0,'gray'); % BPSK 0,1
    
    % can't compute SER/BER of coded symbols/bits for the autoencoder here since we are
    % doing joint coding/modulation
    
    % XXX
    
    % now run the Hamming decoder on the hard decisions
    zb = de2bi(zs);
    tmp = reshape(zb',n,Nblocks)'; % Nblocks by n
    data_hat = decode(tmp,n,k,'hamming/binary');  % Nblocks by k
    data_hat_int = bi2de(data_hat); % Nblocks by 1 (integers from 0 to 15 for each block)
    
    hdec_BER(i1) = biterr(data_int,data_hat_int)/Nbits; % BER after decoder
    hdec_BLER(i1) = symerr(data_int,data_hat_int)/Nblocks;  % BLER after hard decision decoder
    
    % SOFT DEMODULATION/DECODING
    % use z here (scaled channel output)

    data_hat_softdec_tmp = zeros(Nblocks,k);  % hamming
    data_hat_ae_softdec_tmp = zeros(Nblocks,k); % autoencoder
    oo = ones(1,2^k); % row of ones
    
    n1 = 1;
    n2 = n;  % always blocks of n symbols
    i2 = 0;
    while n2<=length(z) % z and ae_z should be the same length
        
        i2 = i2 + 1;
        
        % hamming
        t = z(n1:n2); % needs to be a column (n by 1)
        d = vecnorm(mfbank.' - t*oo);
        [~,imax] = min(d); % imax is the index of the closest codeword in the mfbank
        data_hat_softdec_tmp(i2,:) = double(B8(imax,:)); % k bits (B8 is int8 format)
        
        % autoencoder
        t = ae_z(n1:n2); % needs to be a column (n by 1)
        d = vecnorm(ae_mfbank.' - t*oo);
        [~,imax] = min(d); % imax is the index of the closest codeword in the mfbank
        data_hat_ae_softdec_tmp(i2,:) = double(B8(imax,:)); % k bits (B8 is int8 format)
        
        n1 = n1 + n;
        n2 = n2 + n;
        
    end
    
    % hamming
    data_hat_softdec = reshape(data_hat_softdec_tmp',k,Nblocks)'; %Nblocks by k
    data_hat_softdec_int = bi2de(data_hat_softdec); % Nblocks by 1 (integers from 0 to 15 for each block)
    sdec_BER(i1) = biterr(data_int,data_hat_softdec_int)/Nbits; % BER after decoder
    sdec_BLER(i1) = symerr(data_int,data_hat_softdec_int)/Nblocks;  % BLER after hard decision decoder
    
    % autoencoder
    data_hat_ae_softdec = reshape(data_hat_ae_softdec_tmp',k,Nblocks)'; %Nblocks by k
    data_hat_ae_softdec_int = bi2de(data_hat_ae_softdec); % Nblocks by 1 (integers from 0 to 15 for each block)
    sdec_ae_BER(i1) = biterr(data_int,data_hat_ae_softdec_int)/Nbits; % BER after decoder
    sdec_ae_BLER(i1) = symerr(data_int,data_hat_ae_softdec_int)/Nblocks;  % BLER after hard decision decoder
    
    fprintf('Coded simulation progress  : EbN0 (dB) %4.1f, AE BLER %1.6f, time %6.1f\n',EbN0_dB, sdec_ae_BLER(i1), toc);
    
end


    

% =======================================
% PLOTS
% =======================================

figure(1) % UNCODED SER
semilogy(EbN0_dB_test,unc_SER,'LineWidth',2);
hold on
ax = gca;
ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,unc_SER_theory,'--','Linewidth',2);
hold off
legend('BPSK (sim)','BPSK (theory)');
xlabel('Eb/N0 dB');
ylabel('Uncoded SER');
grid on

figure(2) % UNCODED BER
semilogy(EbN0_dB_test,unc_BER,'LineWidth',2)
hold on
ax = gca;
ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,unc_BER_theory,'--','Linewidth',2);
hold off
legend('BPSK (sim)','BPSK (theory)');
xlabel('Eb/N0 dB');
ylabel('Uncoded BER');
grid on

figure(3) % CODED SER (only for Hamming)
semilogy(EbN0_dB_test,cod_SER,'LineWidth',2);
hold on
ax = gca;
ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,cod_SER_theory,'--','Linewidth',2);
hold off
legend('BPSK (sim)','BPSK (theory)');
xlabel('Eb/N0 dB');
ylabel('Coded SER (before decoder)');
grid on

figure(4) % CODED BER (only for Hamming)
semilogy(EbN0_dB_test,cod_BER,'LineWidth',2);
hold on
ax = gca;
ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,cod_BER_theory,'--','Linewidth',2);
hold off
legend('BPSK (sim)','BPSK (theory)');
xlabel('Eb/N0 dB');
ylabel('Coded BER (before decoder)');
grid on

figure(5) % CODED BER after decoder
semilogy(EbN0_dB_test,hdec_BER,'LineWidth',2);
legend('BPSK');
hold on
ax = gca;
ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,sdec_BER,':','Linewidth',2);
ax = gca;
ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,sdec_ae_BER,'-d','Linewidth',2);
hold off
xlabel('Eb/N0 dB');
ylabel('Coded BER (after decoder)');
title('solid lines are hard decisions, dotted lines are soft decisions');
grid on

%%
figure(6) % BLER after hard/soft decision decoder
semilogy(EbN0_dB_test,hdec_BLER,'LineWidth',2);
hold on
%ax = gca;
%ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,hdec_BLER_theory,'--','Linewidth',2);
%ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,sdec_BLER,':','LineWidth',2);
%ax.ColorOrderIndex = 1;
semilogy(EbN0_dB_test,sdec_ae_BLER,'-d','Linewidth',2);
semilogy(EbN0_dB_test,rcu_BLER_theory,'--','Linewidth',2);
semilogy(EbN0_dB_test,na_BLER_theory,'--','Linewidth',2);
semilogy(EbN0_dB_test,mc_BLER_theory,'--','Linewidth',2);
legend('(7,4) Hamming hard decisions (sim)', '(7,4) Hamming hard decisions (theory)', ...
       '(7,4) Hamming soft decisions (sim)','(7,4) autoencoder (sim)',...
       'RCU bound (theory)','normal approximation (theory)','metaconverse bound (theory)');
hold off
xlabel('Eb/N0 dB');
ylabel('Coded BLER (after decoder)');
%title('solid lines are hard decisions, dotted lines are soft decisions');
grid on
axis([min(EbN0_dB_test) max(EbN0_dB_test) 1E-7 1])