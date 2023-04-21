% gpt2-xl/gpt2-xl_layer_23.mat

load('/home/alfred/Projects/Matlab/LLM/gpt2-xl/gpt2-xl_layer_00.mat')
Q = weights_layer_00.attn_c_attn_weight(1:1600,:);
K = weights_layer_00.attn_c_attn_weight(1601:3200,:);
UU = zeros(size(Q));
N = 1600/64;
Q1 = zeros(N * 1600, N * 64);
K1 = zeros(N * 1600, N * 64);
for i = 1:(1600 / 64)
    Qi = Q((i - 1) * 64 + 1:i *64,:)';
    Ki = K((i - 1) * 64 + 1:i *64,:)';
    Q1((i - 1) * 1600 + 1:i * 1600,(i - 1) * 64 + 1:i * 64) = Qi;
    K1((i - 1) * 1600 + 1:i * 1600,(i - 1) * 64 + 1:i * 64) = Ki;

end

[U,S,V] = svd(Q1*K1');
a = diag(S);
m = sum(a);
b = cumsum(a);
plot(b/m)