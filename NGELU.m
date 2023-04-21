function [y] = NGELU(x)

y = 0.5 .* x .*(1 + tanh(sqrt(2/pi) .* (x + 0.044715 * x.^3)));

end