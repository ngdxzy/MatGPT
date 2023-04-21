function [out] = TokenEmbedding(LUT, in)
% function TokenEmbedding
% 'Embedding' simply means look up table
% LUT: a look up table
% in : token to translate, 1 * N array
% out: output array (features), N * d_model array

in = fix(in); % make interger

out = LUT(in,:); % look up table

end