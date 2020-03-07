% Matched filter generator for soft decision Hamming code simulator
% DRB Mar 4, 2019
% =================================================
% USER PARAMETERS BELOW
% =================================================
mod_order_test = [1]; % 1=BPSK, 2=QPSK, 3=8PSK, 4=16QAM, 6=64QAM
n = 15;
k = 11;
% =================================================

i0 = 0;
for mod_order = mod_order_test
    i0 = i0 + 1;
    
    % make the matched filter bank for soft decisions
    % Matlab encode function only works on blocks of k, so we have to munge
    % the blocks around a bit to get this to work
    index = 0:2^(k*mod_order)-1;  % "superblock" indices
    B = de2bi(index'); % binary matrix containing all possible bit patterns
    enc_B = zeros(2^(k*mod_order),mod_order*n);  % encoded bits
    k1 = 1;
    k2 = k;
    n1 = 1;
    n2 = n;
    while k2<=(mod_order*k)
        enc_B(:,n1:n2) = encode(B(:,k1:k2),n,k,'hamming/binary'); % encoded
        k1 = k1 + k;
        k2 = k2 + k;
        n1 = n1 + n;
        n2 = n2 + n;
    end
    
    % now convert to decimal for modulation -> mfbank
    enc_B_dec = zeros(2^(k*mod_order),n);
    j1 = 1;
    j2 = mod_order;
    m1 = 1;
    while j2<=(mod_order*n)
        enc_B_dec(:,m1) = bi2de(enc_B(:,j1:j2));
        j1 = j1 + mod_order;
        j2 = j2 + mod_order;
        m1 = m1 + 1;
    end
    
    % modulation
    switch mod_order
        case 1
            % BPSK
            mfbank = pskmod(enc_B_dec,2,0,'gray');
            B8 = int8(B);
            save('mfbank_BPSK_15_11.mat','mfbank','B8');
        case 2
            % QPSK
            mfbank = pskmod(enc_B_dec,4,0,'gray');
            B8 = int8(B);
            save('mfbank_QPSK.mat','mfbank','B8');
        case 3
            % 8PSK
            mfbank = pskmod(enc_B_dec,8,0,'gray');
            B8 = int8(B);
            save('mfbank_8PSK.mat','mfbank','B8');            
        case 4
            % 16QAM
            mfbank = qammod(enc_B_dec,16,'UnitAveragePower',true);
            B8 = int8(B);
            save('mfbank_16QAM.mat','mfbank','B8');
        case 6
            % 64QAM
            mfbank = qammod(enc_B_dec,64,'UnitAveragePower',true);
            B8 = int8(B);
            save('mfbank_64QAM.mat','mfbank','B8');
    end
    
end
