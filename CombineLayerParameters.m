% Convert pytorch weights to Matlab weights
modelName = "gpt2-xl";

Layers = 48;
n_model = 1600;
d_k = 64;
heads = n_model / d_k;

for layer = 0:Layers-1
    [~, files] = system("ls " + modelName + " | grep mat | grep h_" + num2str(fix(layer)) + " | sed 's/\.mat//g' | sed 's/h_[0-9]\+_//' ");
    files = regexp(files,"\n","split");
    files = convertCharsToStrings(files(1:end-1));
    weight_name = sprintf("weights_layer_%02d", layer);
    for i = 1:length(files)
        variable_name = "h_" + num2str(fix(layer)) + "_" +files(i);
        load(modelName + "/" + variable_name + ".mat");
        temp = sprintf("%s.%s = %s", weight_name, files(i), variable_name);
        eval(temp)
    end
    filename = sprintf("%s_layer_%02d.mat", modelName,layer);
    save(modelName + "/" + filename,weight_name);

end
