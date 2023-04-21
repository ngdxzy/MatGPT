function [out] = Block(Weights, in)
% function Block
% it is a transformer block
% Weights: A structure containing all weights:
% 
%                                      ▲
%                                      │    out
%                                   ┌──┴──┐
%                                   │  +  ◄────────┐
%                                   └──▲──┘        │
%                                      │           │
%                              ┌───────┴──────┐    │
%          ┌───────────────────► Linear Layer │    │
%          │  mlp_c_proj_weight└───────▲──────┘    │
%          │  mlp_c_proj_bias          │           │
%          │                           │           │
%          │                      ┌────┴────┐      │
%          │                      │ NewRELU │      │
%          │                      └────▲────┘      │    temp
%          │                           │           │
%          │                   ┌───────┴──────┐    │
%          ├───────────────────► Linear Layer │    │
%          │  mlp_c_fc_weight  └───────▲──────┘    │
%          │  mlp_c_fc_bias            │           │
%          │                   ┌───────┴──────┐    │
%          ├───────────────────►  Layer Norm  │    │
%          │  ln_2_weights     └───────▲──────┘    │
%          │  ln_2_bias                │           │
%          │                           ├───────────┘
%          │                           │
%          │                        ┌──┴──┐
%  ────────┤                        │  +  ◄─────────────┐
%  Weights │  attn_c_proj_weight    └──▲──┘             │
%          │  attn_c_proj_bias         │                │
%          │                 ┌─────────┴────────┐       │
%          ├─────────────────►  Self Attention  │       │
%          │                 └─────────▲────────┘       │
%          │  attn_c_attn_weight       │                │   temp
%          │  attn_c_attn_bias         │                │
%          │                   ┌───────┴──────┐         │
%          └───────────────────►  Layer Norm  │         │
%             ln_1_weights     └───────▲──────┘         │
%             ln_1_bias                │                │
%                                      ├────────────────┘
%                                      │ in
% 

% Save temp for future sum
temp = in;

% LayerNorm 
in = LayerNorm(Weights.ln_1_weight, Weights.ln_1_bias, in);
% Do self-attention
in = SelfAttention(Weights, in);
% add together
in = temp + in;

% Save temp for future sum
temp = in;
% LayerNorm 
in = LayerNorm(Weights.ln_2_weight, Weights.ln_2_bias, in);

% Fully connected layer 1: d_model to 4*d_model
% in     : N * d_model
% *weight: (4*d_model) * d_model
% *bias  : 1 * (4*d_model)

in = in * Weights.mlp_c_fc_weight' + Weights.mlp_c_fc_bias;
% Activation function
in = NGELU(in);

% Fully connected layer 2: 4*d_model to d_model
% in     : N * (4 * d_model)
% *weight: d_model * (4*d_model)
% *bias  : 1 * d_model
in = in * Weights.mlp_c_proj_weight' + Weights.mlp_c_proj_bias;

% add
out = in + temp;

% Dropout is not required in forwarding
% out = Dropout(in, 0.1);

end