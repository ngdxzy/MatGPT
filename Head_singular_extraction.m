clear
modelName = "gpt2-base";
[Layers, d_model, dk, n_head] = Get_model_parameters(modelName);


singular_values = zeros(Layers * n_head, dk);

for i = 1:Layers
    fileName = sprintf("%s/%s_layer_%02d.mat", modelName,modelName, i - 1);
    load(fileName);
    valName = sprintf("weights_layer_%02d", i - 1);
    weightName = valName + ".attn_c_attn_weight";
    W = eval(weightName);
    WQ = W(1:d_model,:);
    WK = W(d_model + 1:d_model * 2,:);
    for j = 1:n_head
        wq = WQ((j - 1) * dk + 1:j * dk,:);
        wk = WK((j - 1) * dk + 1:j * dk,:);
        [~,S,~] = svd(wq'*wk);
        singular_values((i - 1) * n_head + j,:) = diag(S(1:dk, 1:dk));
    end
    clear -regexp weights_layer
end