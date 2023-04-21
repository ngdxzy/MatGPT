function [out] = SelfAttention(Weights, in)
% function self attention
% Weights: a set of weights
% in     : N * d_model
% Theory :
%   1. 'in' is copied to Q == K == V;
%   2. transformer uses a special number of heads (Nh) and 
%      inside dimention d_k that follows dk * Nh = d_model;
%   3. each head has a linear layer Wi ad bi, for Q, K and V;
%   4. each head has to to a computation Qi = Q * Wi' + bi;
%   5. to simplify computation, transformer first combins Wi and bi
%            Q  x  [Wq1 Wq2 Wq3 ... WqNh]
%                                        d_model x d_model 
%               + [bq1 bq2 bq3 ... bqNh]
%                                       d_model x 1
%         
%         =  [(QxWq1 + bq1) (QxWq2 + bq2)
%         
%                           ... (QxWqNh + bqNh)]
%                                               d_model x d_model
%       
%                               Wqi:  Weights Qi, d_model x dk
%                               bqi:  Bias Qi, dk x 1
%       The computation becomes Q * Wq + bq, all computation can be done
%      in a single matrix multiplication
%
%   6. Since Q == K == V, transformer further combines Wq, Wk, Wv
%           Q x [Wq Wk Wv] + [bq bk bv] = [QxWq QxWk QxWv]
%                        d_model x (3 x d_model)
%       The computation finally becomes Q x W + b, which can be taken
%      as a single linear layer from d_model to 3xd_model. Notice that
%      the weights are save in a transposed version.

% Get number of tokens and d_model
[N, d_model] = size(in);
% dk is always 64
dk=64;
% number of heads is d_model / dk
n_heads = d_model / dk;

% do all QxWi, QxWk, QxWv together with a single linear layer.
attributions = in * Weights.attn_c_attn_weight' + Weights.attn_c_attn_bias;

% attributions structure:
% 
%            N x d_model
%       ┌─────────────┐
%       │             │
%      [ Q1 Q2 ... QNh    K1 K2 ... KNh V1 V2 ... VNh]
%         ▲
%         │
%         └─ N x dk

% seperate Q, K, V
Q = attributions(:,1:d_model);
K = attributions(:,d_model+1:2*d_model);
V = attributions(:,2*d_model+1:end);

% create a mask
%         ┌─            ─┐
%         │ 1 0 0 0 ... 0│
%         │ 1 1 0 0 ... 0│
%         │ 1 1 1 0 ... 0│
%         │ . . .  .    .│
%         │ . . .    .  .│
%         │ . . .      ..│
%         │ 1 1 1 1 ... 1│
%         └─            ─┘

mask = ones(N);
mask = triu(mask)';

% The final output concats all head output together, the dimention
% is N x d_model again, same with the input
att = zeros(N, d_model);

% for each head
for i = 1:n_heads
    % extract useful data
    Qi = Q(:, (i - 1) * dk + 1:i*dk); % N * dk
    Ki = K(:, (i - 1) * dk + 1:i*dk)'; % dk * N
    Vi = V(:, (i - 1) * dk + 1:i*dk); % N * dk

    local_att = Qi * Ki / sqrt(dk); % N * N
    
    local_att(mask == 0) = -inf;
    % softmax each row, matlab softmax is done on each column, so we have
    % to transpose twice
    local_att = softmax(local_att')'; % N * N

    % Dropout is not required in forwarding
    %  local_att = Dropout(local_att, 0.1);
    % times V
    att(:,(i - 1) * dk + 1:i*dk) = local_att * Vi;
end

% output linear layer
out = att * Weights.attn_c_proj_weight' + Weights.attn_c_proj_bias;
% Dropout is not required in forwarding
%out = Dropout(out, 0.1);

end