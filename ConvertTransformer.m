clear
modelName = "gpt2-base";
[Layers, d_model, dk, n_head] = Get_model_parameters(modelName);

for i = 1:Layers
    fileName = sprintf("%s/layer_%02d.mat",modelName, i - 1);
    load(fileName);
    valName = sprintf("weights_layer_%02d", i - 1);
    weightName = valName + ".attn_c_attn_weight";
    biasName = valName + ".attn_c_attn_bias";
    W = eval(weightName);
    b = eval(biasName);
    WQ = W(1:d_model,:);
    bq = b(1:d_model);
    WK = W(d_model + 1:d_model * 2,:);
    bk = b(d_model + 1:d_model * 2);
    [U,S,V] = svd(WQ'*WK);
    for j = 1:n_head
        wq = WQ((j - 1) * dk + 1:j * dk,:);
        wk = WK((j - 1) * dk + 1:j * dk,:);
        bqs = bq((j - 1) * dk + 1:j * dk);
        bks = bk((j - 1) * dk + 1:j * dk);
        [WQ1, WK1, bq1, bk1, S] = Convert_Model(wq,wk,bqs,bks, 0.8);
        WQ((j - 1) * dk + 1:j * dk,:) = WQ1;
        WK((j - 1) * dk + 1:j * dk,:) = WK1;
        bq((j - 1) * dk + 1:j * dk) = bq1;
        bk((j - 1) * dk + 1:j * dk) = bk1;
    end
    
    W(1:d_model,:) = WQ;
    b(1:d_model) = bq;
    W(d_model + 1:d_model * 2,:) = WK;
    b(d_model + 1:d_model * 2) = bk;
    eval(weightName + " = W;");
    eval(biasName + " = b;");
    fileName = sprintf("%s-mod/layer_%02d.mat", modelName,i - 1);
    save(fileName,valName);
    clear -regexp weights_layer
end