clear
modelName = "gpt2-base";
[Layers, d_model, dk, n_head] = Get_model_parameters(modelName);


singular_values = zeros(Layers, d_model);

for i = 1:Layers
    fileName = sprintf("%s/%s_layer_%02d.mat", modelName,modelName, i - 1);
    load(fileName);
    valName = sprintf("weights_layer_%02d", i - 1);
    weightName = valName + ".attn_c_attn_weight";
    W = eval(weightName);
    WQ = W(1:d_model,:);
    WK = W(d_model + 1:d_model * 2,:);
    [~,S,~] = svd(WK);
    singular_values(i,:) = diag(S);
    clear -regexp weights_layer
end
plot_singular_dist(singular_values)