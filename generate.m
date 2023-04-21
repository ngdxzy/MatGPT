% The GPT tokens is counted from 0, but matlab counts from 1,
% so the input shuold be added by 1
Script_input = inputTokens + 1;

% Token embedding
coded = TokenEmbedding(wte_weight, Script_input);

% Positional embedding
poscode = PostionEmbedding(wpe_weight, length(Script_input));

% transformer input is Token Embedding plus Postional Embedding
final_input = coded + poscode;
clear coded
clear poscode

% Decoder: a series of transformers, each has it's own weights
temp = final_input;
for i = 1:NUM_LAYERS
    % find the weight variable name
    weight_name = sprintf("weights_layer_%02d", i - 1);
    % do transformer block
    temp = Block(eval(weight_name), temp);
end

temp = LayerNorm(ln_f_weight, ln_f_bias, temp);
final_output = temp * lm_head_weight';
logi = final_output(end, :);
%%
prob = softmax(logi');
[~, idx] = max(prob);
Script_output = idx - 1;