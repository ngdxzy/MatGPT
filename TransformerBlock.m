classdef TransformerBlock < handle
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

    properties
        dm
        dk
        Nh
        attn_w
        attn_b
        ln1_weight
        ln1_bias
        attn_proj_w
        attn_proj_b
        ln2_weight
        ln2_bias
        fc1_weight
        fc1_bias
        fc2_weight
        fc2_bias
        
        K_local
        V_local

        last_N
    end

    methods
        function obj = TransformerBlock(dm, dk, Weights)
            obj.dm = dm;
            obj.dk = dk;
            obj.Nh = dm / dk;
            obj.ln1_weight = Weights.ln_1_weight;
            obj.ln1_bias = Weights.ln_1_bias;
            obj.attn_w = Weights.attn_c_attn_weight;
            obj.attn_b = Weights.attn_c_attn_bias;
            obj.attn_proj_w = Weights.attn_c_proj_weight;
            obj.attn_proj_b = Weights.attn_c_proj_bias;
            obj.ln2_weight = Weights.ln_2_weight;
            obj.ln2_bias = Weights.ln_2_bias;
            obj.fc1_weight = Weights.mlp_c_fc_weight;
            obj.fc1_bias = Weights.mlp_c_fc_bias;
            obj.fc2_weight = Weights.mlp_c_proj_weight;
            obj.fc2_bias = Weights.mlp_c_proj_bias;
            
            obj.last_N = 0;
            obj.K_local = zeros(64, dm);
            obj.V_local = zeros(64, dm);

        end

        function out = forward(obj, new_token)
            obj.last_N = obj.last_N + 1;
            temp = LayerNorm(obj.ln1_weight, obj.ln1_bias, new_token);
            temp = temp * obj.attn_w' + obj.attn_b;
            Q = temp(1, 1:obj.dm);
            obj.K_local(obj.last_N, : ) = temp(1, obj.dm * 1 + 1:obj.dm * 2);
            
            obj.V_local(obj.last_N, : ) = temp(1, obj.dm * 2 + 1:obj.dm * 3);

            attn_out = zeros(1, obj.dm);

            for head = 1:obj.Nh
                Qi = Q(1, (head - 1) * obj.dk + 1 : head * obj.dk);
                Ki = obj.K_local(1: obj.last_N, (head - 1) * obj.dk + 1 : head * obj.dk);
                Vi = obj.V_local(1: obj.last_N, (head - 1) * obj.dk + 1 : head * obj.dk);

                temp_att = Qi * Ki' / sqrt(obj.dk);

                temp_att = softmax(temp_att')';

                attn_out(1, (head - 1) * obj.dk + 1 : head * obj.dk) = temp_att * Vi; 
            end
            
            attn_out = attn_out * obj.attn_proj_w' + obj.attn_proj_b;
            
            first_subblock = attn_out + new_token;

            temp = LayerNorm(obj.ln2_weight, obj.ln2_bias, first_subblock);

            temp = temp * obj.fc1_weight' + obj.fc1_bias;

            temp = NGELU(temp);

            temp = temp * obj.fc2_weight' + obj.fc2_bias;

            out = temp + first_subblock;

        end

        function reset(obj)
            obj.last_N = 0;
            obj.K_local(:,:) = 0;
            obj.V_local(:,:) = 0;
        end
    end

end
