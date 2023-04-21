clear 
clc
model_name = "gpt2-base";

if model_name == "gpt2-base"
    d_model = 768;
    heads = 12;
end

W = load(model_name + '/h_0_attn_c_attn_weight.mat');
W = W.h_0_attn_c_attn_weight;
%%
WQ = W(1:d_model,:);
WK = W(d_model + 1:d_model * 2,:);
WV = W(2 * d_model+1: end,:);

%%
MW = WQ * WK';
[U, S, V] = svd(MW);
