% Pe = rcu(EbN0,n,R)
% inputs:
%   EbN0 = the usual energy per bit / PSD level of noise (not in dB)
%   n    = length of the codeword
%   R    = k/n rate of the code
% outputs: 
%   Pe  = RCU bound on error/BLER probability (equation (61) of Coding in
%   the Finite-Blocklength Regime: Bounds Based on Laplace Integrals and Their Asymptotic Approximations)
%   by Tomaso Erseghe
%
% Needs helper functions: rhopdf.m, rhopdfapprox.m, and weberU.m
%
% Has to use an approximation for n>40

function Pe = rcu(EbN0,n,R)

v = n-1;                                            % degrees of freedom
Omega = 2*R*EbN0;                                   % SNR

if n<=500
    
    % first find lambda so that g(lambda) = 2^(-n*R)
    % we know g(a) = Prob(eta > a) = tcdf(sqrt(v./(1-a.^2)).*a,v,'upper')
    % we could do this with tinv, but I don't know how well it computes tail
    % probabilities, so we'll use fzero instead - this seems to work to at
    % least n=200 and R=1/2
    fun = @(a) -(1/n)*log2(tcdf(sqrt(v./(1-a.^2)).*a,v,'upper'))-R; % function
    lambda = fzero(fun,0);
    
    if n<=40 % do the exact integral numerically
        
        % first integral
        part1 = quadgk(@(a) rhopdf(a,n,Omega),-1,lambda);
        
        % second integral, note: g(lambda) = 2^(-n*R)
        part2 = 2^(n*R)*quadgk(@(a) tcdf(sqrt(v./(1-a.^2)).*a,v,'upper').*rhopdf(a,n,Omega),lambda,1);
        
    else % do an approximate integral numerically
        
        % first integral
        part1 = quadgk(@(a) rhopdfapprox(a,n,Omega),-1,lambda);
        
        % second integral, note: g(lambda) = 2^(-n*R)
        part2 = 2^(n*R)*quadgk(@(a) tcdf(sqrt(v./(1-a.^2)).*a,v,'upper').*rhopdfapprox(a,n,Omega),lambda,1);
    
    end

    % put it all together
    Pe = part1 + part2;
    
else % n>500, integrals don't work, can't find lambda with fzero, so use Theorem 17
    
    lambda = sqrt(1-2^(-2*R)*exp(log(n)/n));  % approximation (eq (75))
    
    alpha = sqrt(Omega/4);
    a = lambda;
    w1 = alpha*a + sqrt(1+(alpha*a)^2);
    c = 2*alpha*(1-a^2)*w1;
    u0 = (1/2)*log(1-a^2) - 2*alpha^2 + ...
        (alpha*a)^2 + (alpha*a)*sqrt(1+(alpha*a)^2) + ...
        log(w1);
    w0 = sqrt(((1-a^2)/a^2)*(1+alpha*a*w1)) * (c-a)*(2*a-c);
     
    tmp = u0 - log(2*pi*n)/(2*n) - (1/n)*log(w0);
    Pe = exp(n*tmp);
        
end

