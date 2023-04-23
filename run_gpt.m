clear
clc
%% Load weight files
modelName = "gpt2-base";

[~, files] = system("ls " + modelName + " | grep mat | sed 's/.mat//'");
%                    show all files | files with mat | delete .mat suffix

% split into an array of file names
files = regexp(files,"\n","split");

% combine characters into string, delete the last one, which is usless
files = convertCharsToStrings(files(1:end-1)); 

for i = 1:length(files)
    load(modelName + "/" + files(i) + ".mat");
end

% define global variables
[NUM_LAYERS, d_model, dk, n_head] = Get_model_parameters(modelName);

clear files
%% Run GPT

Question = "What is your job?";

% define generation steps
Steps = 15;

% input Tokens start with empty array
inputTokens = [];
% initial input is loaded into the 'out' array for identical loop structure
out = bpe_encoding(Question);

% Generation steps
for i = 1:Steps
    % Concat last input with last output
    inputTokens = [inputTokens out];
    % call generate script, it is not a function to avoid passing too many
    % arguments
    generate;

    % load final output into 'out'
    out = Script_output;
end

% Print output
final_out = [inputTokens out];
bpe_decoding(final_out)
