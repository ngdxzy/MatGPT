function [out] = LayerNorm(w, b, in)

m = mean(in,2);
v = var(in,0,2);
v = sqrt(v + 1e-5);

in = in - m;
in = in ./ v;
out = in .* w + b;


end