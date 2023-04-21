function [NUM_LAYERS, d_model, dk, n_head] = Get_model_parameters(modelName)

if(modelName == "gpt2-base" || modelName == "gpt2-base-mod")
    NUM_LAYERS = 12;
    d_model = 768;
    dk = 64;
    n_head = d_model / dk;
else
    if (modelName == "gpt2-large" || modelName == "gpt2-large-mod")
        NUM_LAYERS = 36;
        d_model = 1280;
        dk = 64;
        n_head = d_model / dk;
    else
        NUM_LAYERS = 48;
        d_model = 1600;
        dk = 64;
        n_head = d_model / dk;
    end
end

end