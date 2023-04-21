clear

model_name = "gpt2-base";


Layers = 12;
d_model = 768;
singular_values = zeros(Layers * d_model, 1);

for i = 1:Layers
    fileName = sprintf("gpt2-base/gpt2-base_layer_%02d.mat", i - 1);
    load(fileName);
    valName = sprintf("weights_layer_%02d", i - 1);
    weightName = valName + ".attn_c_attn_weight";
    W = eval(weightName);
    b = eval(biasName);
    WQ = W(1:d_model,:);
    bq = b(1:d_model);
    WK = W(d_model + 1:d_model * 2,:);
    bk = b(d_model + 1:d_model * 2);
    [U,S,V] = svd(WQ'*WK);
    singular_values((i - 1) * d_model + 1:i * d_model) = diag(S);
    %[WQ1, WK1, bq1, bk1, S] = Convert_Model(wq,wk,bqs,bks);
%     for j = 1:(1600/64)
%         wq = WQ((j - 1) * 64 + 1:j * 64,:);
%         wk = WK((j - 1) * 64 + 1:j * 64,:);
%         bqs = bq((j - 1) * 64 + 1:j * 64);
%         bks = bk((j - 1) * 64 + 1:j * 64);
%         [WQ1, WK1, bq1, bk1, S] = Convert_Model(wq,wk,bqs,bks);
%         WQ((j - 1) * 64 + 1:j * 64,:) = WQ1;
%         WK((j - 1) * 64 + 1:j * 64,:) = WK1;
%         bq((j - 1) * 64 + 1:j * 64) = bq1;
%         bk((j - 1) * 64 + 1:j * 64) = bk1;
%         singular_values((i - 1) * 1600 + (j - 1) * 64 + 1:(i - 1) * 1600 + j * 64) = diag(S(1:64,1:64));
%     end
    
    W(1:d_model,:) = WQ;
    b(1:d_model) = bq;
    W(d_model + 1:d_model * 2,:) = WK;
    b(d_model + 1:d_model * 2) = bk;
    eval(weightName + " = W;");
    eval(biasName + " = b;");
    fileName = sprintf("gpt2-base-mod/gpt2-base_mod_layer_%02d.mat", i - 1);
    %save(fileName,valName);
    clear -regexp weights_layer
end