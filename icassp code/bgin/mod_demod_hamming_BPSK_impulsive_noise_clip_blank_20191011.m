% Hamming code simulator
% Hard and soft decoding
% Moved the matched filter bank generator to a separate file
% DRB Mar 4, 2019

% KPV Sept 1, 2019
% Modified for impulsive channel
% Changes :
% 1. Replaced complex noise with impulsive complex noise
% 2. Loop for probabilities from 0:0.1:1 - replace p with pvec(i3)
% 3. Changed the error variables from 2-dim to 3-dim, the 3rd dimension
% being the length the probability vector

clc
clear
close
% =================================================
% USER PARAMETERS BELOW
% =================================================
Nbits = 1E7;% 8E6 total number of bits (must be divisible by all of the modulation orders after multiplication by n/k)

mod_order_test = 1; % 1=BPSK, 2=QPSK, 3=8PSK, 4=16QAM, 6=64QAM
pvec = 0:0.1:1;
prob_string = ["0","0point1","0point2","0point3","0point4","0point5","0point6","0point7","0point8","0point9","1"];
n = 7;
k = 4;
max_mod_order_sdec = 1;  % maximum modulation order for soft decisions

% Impulsive noise parameters
EbN0_dB_1 = 3.0;
EbN0_dB_2 = -7.0;
EbN0_dB_test = EbN0_dB_1;
R = k/n;
EbN0_1 = 10.^(EbN0_dB_1/10);
noise_std_1 = 1/sqrt(2*R*EbN0_1);
EbN0_2 = 10.^(EbN0_dB_2/10);
noise_std_2 = 1/sqrt(2*R*EbN0_2);

Nencbits = Nbits*n/k;
Nblocks = Nbits/k;

% diagnostic variables
unc_Es = zeros(length(mod_order_test),length(EbN0_dB_test), length(pvec));
unc_Eb = zeros(length(mod_order_test),length(EbN0_dB_test), length(pvec)); % 10*log10(unc_Eb/2) should match EbN0_dB_test
cod_Es = zeros(length(mod_order_test),length(EbN0_dB_test), length(pvec));
cod_Eb = zeros(length(mod_order_test),length(EbN0_dB_test), length(pvec)); % 10*log10((n/k)*cod_Eb/2) should match EbN0_dB_test
ae_cod_Es = zeros(length(mod_order_test),length(EbN0_dB_test), length(pvec));
ae_cod_Eb = zeros(length(mod_order_test),length(EbN0_dB_test), length(pvec)); % 10*log10((n/k)*cod_Eb/2) should match EbN0_dB_test

%% =========================================
% UNCODED SIMULATION
% ==========================================
unc_SER = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec));
unc_BER = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec));
data = randi([0 1],Nbits,1); % generate a vector with all N bits
data_int = bi2de(reshape(data,k,Nblocks)'); % need this for BLER

tic
i0 = 0;
for i3 = 1:length(pvec)
    for mod_order = mod_order_test
        i0 = i0+1;        
        Nsyms = Nbits/mod_order;

        i1 = 0;
        for EbN0_dB = EbN0_dB_test
            i1 = i1+1;
            
            EbN0 = 10^(EbN0_dB/10);
            
            % MODULATION (uncoded)
             % BPSK
            x = data;
            s = pskmod(x,2,0,'gray');
            ss = s*sqrt(2*EbN0);  % scale (each symbol has energy k*Eb/n)
            
            % CHANNEL
            %         y = ss + randn(length(s),1) + 1i*randn(length(s),1); % add complex noise
            impulsive_noise = bernoulli_gaussian(noise_std_1, noise_std_2, length(ss), pvec(i3));
            y = ss + impulsive_noise; % add complex noise
            
            % DIAGNOSTIC
            unc_Es(mod_order,i1,i3) = (ss'*ss)/length(ss);
            unc_Eb(mod_order,i1,i3) = unc_Es(mod_order,i1,i3)/mod_order;
            
            % DEMODULATION
            % BPSK
            zs = pskdemod(y,2,0,'gray'); % outputs are already binary
            zb = zs;
            
            unc_SER(mod_order,i1,i3) = symerr(x,zs)/Nsyms; % SER
            unc_BER(mod_order,i1,i3) = biterr(x,zs)/Nbits; % BER
            
            fprintf('Uncoded simulation progress: mod_order %i, EbN0 (dB) %4.1f, prob. %4.1f, time %6.1f\n',mod_order,EbN0_dB,pvec(i3),toc);
            
        end      
    end
end

%%  
%=========================================
% HAMMING CODED SIMULATION (use same data vector as for uncoded)
% ==========================================
cod_SER = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec)); % before decoding
cod_BER = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec)); % before decoding
hdec_BER = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec)); % after decoding
hdec_BLER = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec)); % after decoding
sdec_BER = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec)); % after decoding
sdec_BLER = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec)); % after decoding

hdec_BER_hybrid = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec)); % after decoding
hdec_BLER_hybrid = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec)); % after decoding
sdec_BER_hybrid = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec)); % after decoding
sdec_BLER_hybrid = zeros(length(mod_order_test),length(EbN0_dB_test),length(pvec)); % after decoding

tic
i0 = 0;
for i3 = 1:length(pvec)
    for mod_order = mod_order_test
        i0 = i0+1;
        % mfbank used for both coding/modulation and soft decision decoding
%         load('mfbank_BPSK.mat'); % loads mfbank and B8
        if (k == 4)
            load('mfbank_BPSK_7_4.mat'); % loads mfbank and B8
        elseif (k == 7)
            load('mfbank_BPSK_15_11.mat'); % loads mfbank and B8
        end
%         % With 3 and -7
%         load(strcat('ae_mfbanks_impulsive/ae_mfbank_hamming_impulsive_20190905_1135_bpsk_EbNodB1_',num2str(3),'_EbNodB2_',num2str(-7),'_prob_',prob_string(i3),'.mat')); % loads autoencoder S_encoded_syms_bpsk and bit_dict_bpsk
%         ae_mfbank = S_encoded_syms;
%         clear S_encoded_syms_bpsk;
        
        Nsuperblocks = Nbits/(k*mod_order);
        Nencsyms = Nencbits/mod_order;
        
        % reshape into "superblocks" with one superblock in each row and convert to integer indices
        data_sb_int = bi2de(reshape(data,k*mod_order,Nsuperblocks)');
        
        % Hamming joint coding and modulation (unit average energy per symbol here)
        s = zeros(n*Nsuperblocks,1); % coded/modulated signal
        for i2 = 1:Nsuperblocks
            index = data_sb_int(i2)+1;
            s((i2-1)*n+1:i2*n) = mfbank(index,:);
        end
        
%         % Autoencoder joint coding and modulation (unit average energy per symbol here)
%         ae_s = zeros(n*Nsuperblocks,1); % coded/modulated signal
%         for i2 = 1:Nsuperblocks
%             index = data_sb_int(i2)+1;
%             ae_s((i2-1)*n+1:i2*n) = ae_mfbank(index,:);
%         end
        
        i1 = 0;
        for EbN0_dB = EbN0_dB_test
            i1 = i1+1;
            
            EbN0 = 10^(EbN0_dB/10);
            
            % now scale to proper EbN0 (each symbol has energy Eb*k/n)
%             ss = s*sqrt(mod_order*2*EbN0*k/n);  % Hamming
%             ae_ss = ae_s*sqrt(mod_order*2*EbN0*k/n);  % autoencoder
            ss = s;
%             ae_ss = ae_s;
            % CHANNEL
            impulsive_noise = bernoulli_gaussian(noise_std_1, noise_std_2, length(s), pvec(i3));
            y = s + impulsive_noise; % add complex impulsive noise
%             ae_y = ae_s + impulsive_noise; % add complex impulsive noise
            
            
            % Non-linear pre-processors at receiver for mitigating impulsive noise
            
%             
            %%% CONVENTIONAL CLIPPING 
%             clipping_threshold = noise_std_1*(sqrt(2*log((1-pvec(i3))*sqrt(2)/pvec(i3))));
%             clipping_threshold = pvec(i3);
            clipping_threshold = mean(abs(y));
            y_hybrid = y;
%             for i = 1:length(y_hybrid)
%                 if abs(y_hybrid(i)) > clipping_threshold
%                     y_hybrid(i) = clipping_threshold*exp(1j*angle(y_hybrid(i)*i));
% %                     y_hybrid(i) = 0;
%                 end
%                 if abs(y_hybrid(i)) <= clipping_threshold
%                     y_hybrid(i) = y_hybrid(i);
%                 end
%             end
            y = y_hybrid;
%             
%             %%% BLANKING
            blanking_threshold = 1.5*mean(abs(y));
             y_hybrid = y;
            for i = 1:length(y_hybrid)
                if abs(y_hybrid(i)) > blanking_threshold
                    y_hybrid(i) = 0;
                end
                if abs(y_hybrid(i)) <= blanking_threshold
                    y_hybrid(i) = y_hybrid(i);
                end
            end
            y = y_hybrid;
          

%             %%% Hybrid 
            clipping_threshold = mean(abs(y));
            blanking_threshold = 1.5*mean(abs(y));
            y_hybrid = y;
%             for i = 1:length(y_hybrid)
%                 if abs(y_hybrid(i)) <= clipping_threshold
%                     y_hybrid(i) = y_hybrid(i);
%                 end
%                 if abs(y_hybrid(i)) > clipping_threshold && abs(y_hybrid(i)) <= blanking_threshold
%                     y_hybrid(i) = clipping_threshold*exp(1j*angle(y_hybrid(i)*i));
%                 end
%                 if abs(y_hybrid(i)) > blanking_threshold
%                     y_hybrid(i) = 0;
%                 end
%             end
             y = y_hybrid;
            
            % DIAGNOSTIC
            cod_Es(mod_order,i1,i3) = (ss'*ss)/length(ss);
            cod_Eb(mod_order,i1,i3) = cod_Es(mod_order,i1,i3)/mod_order;

            
            % HARD DECISION DEMODULATION (ONLY FOR HAMMING CODE)
            z = y/sqrt(mod_order*2*EbN0*k/n); % scale
            z_hybrid = y_hybrid/sqrt(mod_order*2*EbN0*k/n); % scale

            
            zs = pskdemod(z,2,0,'gray'); % BPSK 0,1
            zs_hybrid = pskdemod(z_hybrid,2,0,'gray'); % BPSK 0,1
            
            % now run the Hamming decoder on the hard decisions
            zb = de2bi(zs);
            tmp = reshape(zb',n,Nblocks)'; % Nblocks by n
            
            zb_hybrid = de2bi(zs_hybrid);
            tmp_hybrid = reshape(zb_hybrid',n,Nblocks)'; % Nblocks by n
            
            
            data_hat = decode(tmp,n,k,'hamming/binary');  % Nblocks by k
            data_hat_int = bi2de(data_hat); % Nblocks by 1 (integers from 0 to 15 for each block)
            
            data_hat_hybrid = decode(tmp_hybrid,n,k,'hamming/binary');  % Nblocks by k
            data_hat_int_hybrid = bi2de(data_hat_hybrid); % Nblocks by 1 (integers from 0 to 15 for each block)
            
            hdec_BER(mod_order,i1,i3) = biterr(data_int,data_hat_int)/Nbits; % BER after decoder
            hdec_BLER(mod_order,i1,i3) = symerr(data_int,data_hat_int)/Nblocks;  % BLER after hard decision decoder
            
            hdec_BER_hybrid(mod_order,i1,i3) = biterr(data_int,data_hat_int_hybrid)/Nbits; % BER after decoder
            hdec_BLER_hybrid(mod_order,i1,i3) = symerr(data_int,data_hat_int_hybrid)/Nblocks;  % BLER after hard decision decoder
            
          
            if mod_order <= max_mod_order_sdec
                
                data_hat_softdec_tmp = zeros(Nblocks/mod_order,k*mod_order);  % hamming
                data_hat_ae_softdec_tmp = zeros(Nblocks/mod_order,k*mod_order); % autoencoder
                oo = ones(1,2^(k*mod_order)); % row of ones
                
                n1 = 1;
                n2 = n;  % always blocks of n symbols
                i2 = 0;
                while n2<=length(z) % z and ae_z should be the same length
                    
                    i2 = i2 + 1;
                    
                    % hamming
                    t = z(n1:n2); % needs to be a column (n by 1)
                    d = vecnorm(mfbank.' - t*oo);
                    [~,imax] = min(d); % imax is the index of the closest codeword in the mfbank
                    data_hat_softdec_tmp(i2,:) = double(B8(imax,:)); % mod_order*k bits (B8 is int8 format)
                    
%                     % autoencoder
%                     t = ae_z(n1:n2); % needs to be a column (n by 1)
%                     d = vecnorm(ae_mfbank.' - t*oo);
%                     [~,imax] = min(d); % imax is the index of the closest codeword in the mfbank
%                     data_hat_ae_softdec_tmp(i2,:) = double(B8(imax,:)); % mod_order*k bits (B8 is int8 format)
%                     
                    n1 = n1 + n;
                    n2 = n2 + n;
                    
                end
                
                % hamming
                data_hat_softdec = reshape(data_hat_softdec_tmp',k,Nblocks)'; %Nblocks by k
                data_hat_softdec_int = bi2de(data_hat_softdec); % Nblocks by 1 (integers from 0 to 15 for each block)
                sdec_BER(mod_order,i1,i3) = biterr(data_int,data_hat_softdec_int)/Nbits; % BER after decoder
                sdec_BLER(mod_order,i1,i3) = symerr(data_int,data_hat_softdec_int)/Nblocks;  % BLER after hard decision decoder
                
             
            end
            
            fprintf('Coded simulation progress  : mod_order %i, EbN0 (dB) %4.1f, prob. %4.1f, time %6.1f\n',mod_order,EbN0_dB,pvec(i3),toc); 
        end    
    end
end

%%
hdec_BLER = squeeze(hdec_BLER);
sdec_BLER = squeeze(sdec_BLER);

%% 
load('ae_hamming_AWGN_bler_results_7_4_bpsk_EbNodB1_3_EbNodB2_-7.mat')  % Load AE result

hdec_BLER_AWGN = load('results/hdec_BLER_hamming_AWGN.mat','hdec_BLER'); % Load sdec_BLER_AWG
hdec_BLER_AWGN = hdec_BLER_AWGN.hdec_BLER;
sdec_BLER_AWGN = load('results/sdec_BLER_hamming_AWGN.mat','sdec_BLER'); % Load sdec_BLER_AWG
sdec_BLER_AWGN = sdec_BLER_AWGN.sdec_BLER;

hdec_BLER_AWGN_clipping = load('results/hdec_BLER_hamming_AWGN_clipping.mat','hdec_BLER'); % Load sdec_BLER_AWG
hdec_BLER_AWGN_clipping = hdec_BLER_AWGN_clipping.hdec_BLER;
sdec_BLER_AWGN_clipping = load('results/sdec_BLER_hamming_AWGN_clipping.mat','sdec_BLER'); % Load sdec_BLER_AWG
sdec_BLER_AWGN_clipping = sdec_BLER_AWGN_clipping.sdec_BLER;

hdec_BLER_AWGN_blanking = load('results/hdec_BLER_hamming_AWGN_blanking.mat','hdec_BLER'); % Load sdec_BLER_AWG
hdec_BLER_AWGN_blanking = hdec_BLER_AWGN_blanking.hdec_BLER;
sdec_BLER_AWGN_blanking = load('results/sdec_BLER_hamming_AWGN_blanking.mat','sdec_BLER'); % Load sdec_BLER_AWG
sdec_BLER_AWGN_blanking = sdec_BLER_AWGN_blanking.sdec_BLER;

hdec_BLER_AWGN_hybrid = load('results/hdec_BLER_hamming_AWGN_hybrid.mat','hdec_BLER'); % Load sdec_BLER_AWG
hdec_BLER_AWGN_hybrid = hdec_BLER_AWGN_hybrid.hdec_BLER;
sdec_BLER_AWGN_hybrid = load('results/sdec_BLER_hamming_AWGN_hybrid.mat','sdec_BLER'); % Load sdec_BLER_AWG
sdec_BLER_AWGN_hybrid = sdec_BLER_AWGN_hybrid.sdec_BLER;

EbN0_dB_1 = 3.0;
EbN0_dB_2 = -7.0;
figure;
semilogy(pvec, hdec_BLER_AWGN(1)*ones(size(pvec)),'m-.','LineWidth',2)
hold on
semilogy(pvec, sdec_BLER_AWGN(1)*ones(size(pvec)),'c-.','LineWidth',2);
semilogy(pvec, hdec_BLER_AWGN(end)*ones(size(pvec)),'m--','LineWidth',2);
semilogy(pvec, sdec_BLER_AWGN(end)*ones(size(pvec)),'c--','LineWidth',2);
semilogy(pvec,hdec_BLER_AWGN,'b','LineWidth',2);
semilogy(pvec,sdec_BLER_AWGN,'-ok','LineWidth',2);
semilogy(pvec,hdec_BLER_AWGN_hybrid,'-bd','LineWidth',2);
semilogy(pvec,sdec_BLER_AWGN_hybrid,'-kd','LineWidth',2);
semilogy(pvec,hdec_BLER_AWGN_clipping,'-bs','LineWidth',2);
semilogy(pvec,sdec_BLER_AWGN_clipping,'-ks','LineWidth',2);
semilogy(pvec,hdec_BLER,'-bx','LineWidth',2);
semilogy(pvec,sdec_BLER,'-kx','LineWidth',2);
semilogy(pvec,ae_BLER,':rd','LineWidth',2);
legend('HDD at high EbN0',...
       'SDD at high EbN0','HDD at low EbN0',...
       'SDD at low EbN0','BPSK h74 hard','BPSK h74 soft',...
       'BPSK h74 hybrid hard','BPSK h74 hybrid soft',...
       'BPSK h74 clipping hard','BPSK h74 clipping soft',...
       'BPSK h74 blanking hard','BPSK h74 blanking soft',...
       'BPSK ae','Location','best');
hold off
xlabel('probability');
ylabel('Block Error Rate');
grid on
saveas(gcf, strcat('figs/with_clip_blank_hyb_', num2str(EbN0_dB_1),'_and_', num2str(EbN0_dB_2),'.eps'));
saveas(gcf, strcat('figs/with_clip_blank_hyb_', num2str(EbN0_dB_1),'_and_', num2str(EbN0_dB_2),'.fig'));