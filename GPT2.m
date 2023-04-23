classdef GPT2 < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties
        NUM_LAYERS
        d_model
        dk
        n_head
    
        wte_weight
        wpe_weight
        ln_f_weight
        ln_f_bias
        lm_head_weight

        Blocks
    end

    methods
        function obj = GPT2(modelName)
            [obj.NUM_LAYERS, obj.d_model, obj.dk, obj.n_head] = Get_model_parameters(modelName);

            data = load(modelName + "/wte_weight.mat");
            obj.wte_weight = data.wte_weight;

            data = load(modelName + "/wpe_weight.mat");
            obj.wpe_weight = data.wpe_weight;

            data = load(modelName + "/ln_f_weight.mat");
            obj.ln_f_weight = data.ln_f_weight;

            data = load(modelName + "/ln_f_bias.mat");
            obj.ln_f_bias = data.ln_f_bias;

            data = load(modelName + "/lm_head_weight.mat");
            obj.lm_head_weight = data.lm_head_weight;

            obj.Blocks = {};
            for layer = 1:obj.NUM_LAYERS
                data = load(sprintf("%s/layer_%02d.mat",modelName, layer - 1));
                eval(sprintf("obj.Blocks{layer} = TransformerBlock(obj.d_model, obj.dk, data.weights_layer_%02d);", layer - 1));
            end
            clear data
        end

        function Answer = generate(obj, Question, Steps)
            for layer = 1:obj.NUM_LAYERS
                obj.Blocks{layer}.reset();
            end
            
            % initial input is loaded into the 'out' array for identical loop structure
            inputTokens = bpe_encoding(Question);
            
            Pos = 1;
            % initialize transformer
            for Aski = 1:length(inputTokens)
                inputToken = inputTokens(Aski);        
                loop_output = obj.forward(inputToken, Pos);
                Pos = Pos + 1;
            end
            
            new_out = loop_output;
            % Generation steps
            Answer = zeros(1, Steps);
            for Ansi = 1:Steps
                % Concat last input with last output
                inputToken = new_out;
                loop_output = obj.forward(inputToken, Pos);
                Answer(Ansi) = loop_output;
                new_out = loop_output;
            
                Pos = Pos + 1;
                % load final output into 'out'
            end
            
            % Print output
            Answer = bpe_decoding(Answer);
            Answer = eraseBetween(Answer,1,1);

        end

        function outputToken = forward(obj, inputToken, Pos)
            Loop_input = inputToken + 1;

            coded = obj.wte_weight(Loop_input,:);
            
            % Positional embedding
            poscode = obj.wpe_weight(Pos, :);
            
            % transformer input is Token Embedding plus Postional Embedding
            final_input = coded + poscode;

            temp = final_input;
            for i = 1:obj.NUM_LAYERS
                temp = obj.Blocks{i}.forward(temp);
            end
            
            temp = LayerNorm(obj.ln_f_weight, obj.ln_f_bias, temp);
            final_output = temp * obj.lm_head_weight';
            logi = final_output(end, :);
            %%
            prob = softmax(logi');
            [~, idx] = max(prob);
            outputToken = idx - 1;
        end
    end
end