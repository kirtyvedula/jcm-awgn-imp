function [noise] = bernoulli_gaussian(noise_std_1, noise_std_2, N, p)
    
% var_low = 1/(10.^(EbN0_dB_low/10));
% var_high = 1/(10.^(EbN0_dB_high/10));

    q = rand(N,1);
    mask_bad_channel = (q < p); % Pick the bad channel
    mask_good_channel = (q >= p);
%     x1 = sqrt(var_high).*(randn(N,1) + 1i*randn(N,1));  % noise samples where q < p
%     x2 = sqrt(var_low).*(randn(N,1) + 1i*randn(N,1));  % noise samples where q >= p
%     x1 = noise_std_1.*(randn(N,1));  % noise samples where q < p
%     x2 = noise_std_2.*(randn(N,1));  % noise samples where q >= p
%     
    
        x1 = noise_std_1.*(randn(N,1) + 1i*randn(N,1));  % noise samples where q < p
    x2 = noise_std_2.*(randn(N,1) + 1i*randn(N,1));  % noise samples where q >= p
    
    
    noise = mask_good_channel.*x1 + mask_bad_channel.*x2;
%     noise = x1
    
%     % Sanity checks
%     x1_mean = mean(x1); % Almost 0
%     x2_mean = mean(x2); % Almost 0
%     x1_var = var(x1); % Should be equal to var_high - almost equal
%     x2_var = var(x2); % Should be equal to var_low - equal
%     
%     
%     noise_mean = mean(noise);
%     noise_var = var(noise);
%     
%     disp('Over');
%     
end


% 1. compute the mean and variance of the noise samples where q>=p; this
% should be zero and var1  - DONE
% 2. compute the mean and variance of the noise
% samples where q<p; this should be zero and var2 - DONE 
% 3. compute the overall
% unconditional mean and variance; this should be zero and
% p*var2+(1-p)*var1 (assuming p is the probability of getting the bad var2
% channel) - DONE
% 4. compute autocorrelations; this should have a spike at zero
% lag of height p*var2+(1-p)*var1 and then should be close to zero at all
% other lags

