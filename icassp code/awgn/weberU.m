% function y = weberU(a,x)
% Abramowitz and Stegun 19.5.3
% *** only valid when a+1/2 is a positive integer ***

function y = weberU(a,xvec)

y = zeros(size(xvec));

i1 = 0;
for z = xvec
    i1 = i1+1;
    
    part1 = exp(-z^2/4)/gamma(1/2+a);
    fun = @(s) exp(-z*s-s.^2/2).*s.^(a-1/2);  % parameterized function
    part2 = quadgk(fun,0,inf);
    y(i1) = part1*part2;
    
end


