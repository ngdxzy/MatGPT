function [A] = Dropout(A, p)

mask = rand(size(A));
mask = mask >= p;

A = A .* mask;

end