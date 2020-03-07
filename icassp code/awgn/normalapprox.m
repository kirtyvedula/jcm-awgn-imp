% Pe = normalapprox(EbN0,n,R)
% inputs:
%   EbN0 = the usual energy per bit / PSD level of noise (not in dB)
%   n    = length of the codeword
%   R    = k/n rate of the code
% outputs: 
%   Pe  = normal approximation on BLER

function Pe = normalapprox(EbN0,n,R)

Omega = 2*R*EbN0;                                   % SNR
V = Omega*(2+Omega)/(2*(1+Omega)^2);                % dispersion
C = 0.5*log2(1+Omega);                              % capacity
Pe = qfunc(sqrt(n/V)*((C-R)/log2(exp(1))+log(n)/2/n));
