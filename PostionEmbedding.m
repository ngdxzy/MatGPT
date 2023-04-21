function [out] = PostionEmbedding(LUT, N)
% function PostionEmbedding
% 'Embedding' simply means look up table
% LUT: a look up table
% N  : number of tokens
% out: output array (features), N * d_model array

% the positional encoding doesn't look like any sinsoid. GPT trains the
% postional encoding as well.
idx = 1:N;

out = LUT(idx,:);

end