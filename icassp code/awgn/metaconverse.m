% PMD = metaconverse(EbN0,n,R)
% inputs:
%   EbN0 = the usual energy per bit / PSD level of noise (not in dB)
%   n    = length of the codeword
%   R    = k/n rate of the code
% outputs: 
%   PMD  = probability of missed detection which is also equivalent to P_bar_e
%
% All of the calculatiomns here are based on equations (10) and (13) from
% On the Evaluation of the Polyanskiy?Poor?Verdú Converse Bound for Finite Block-Length Coding in AWGN
% by Tomaso Erseghe

function PMD = metaconverse(EbN0,n,R)

Omega = 2*R*EbN0;                                   % SNR
PFA = 2^(-n*R);                                     % probability of false alarm
if n<=200
    t = ncx2inv(PFA,n,n*(1+Omega)/Omega);           % threshold to satisfy PFA constraint (only seems to work for n<=200
else
    fun = @(x) ncx2cdf(x,n,n*(1+Omega)/Omega)-PFA;  % cdf
    x0 = (n + n*(1+Omega)/Omega)/2;                 % half the mean as an initial guess
    t = fzero(fun,x0);
end
lambda_prime = t*(1+Omega)/n;                       % convert threshold to lambda'
PMD = ncx2cdf(n*lambda_prime,n,n/Omega,'upper');    % probability of missed detection
