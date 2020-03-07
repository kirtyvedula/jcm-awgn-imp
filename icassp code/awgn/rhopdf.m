% function y = rhopdf(a,n,Omega)
% from equation (58) of 
% Coding in the Finite-Blocklength Regime:
% Bounds based on Laplace Integrals
% and their Asymptotic Approximations
% Tomaso Erseghe

function y = rhopdf(a,n,Omega)

num = (1-a.^2).^((n-3)/2).*exp(-(n/2)*(1-a.^2/2)*Omega);
den = 2^((n/2)-1)*gamma(1/2)*gamma((n-1)/2)/gamma(n);
y = (num/den).*weberU(n-1/2,-a*sqrt(n*Omega));