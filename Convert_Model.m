function [Q1, K1, bq1, bk1, S] = Convert_Model(Q, K, bq, bk, loss)
[dk, ~] = size(Q);
[U,S,V] = svd(Q'*K);
S = sqrt(S);
ndk = fix(dk * loss);
if (ndk < dk)
    S(ndk + 1:end,ndk + 1:end) = 0;
end
U1 = U*S;
U1 = U1(:, 1:dk);
V1 = S*V';
V1 = V1(1:dk,:);
S = S.^2;
bq1 = bq * (Q'\U1);
bk1 = bk * (K'\V1');
Q1 = U1';
K1 = V1;
end
