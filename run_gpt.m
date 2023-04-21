clear
clc
%% Load weight files
modelName = "gpt2-xl-mod";

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
if(modelName == "gpt2-base" || modelName == "gpt2-base-mod")
    NUM_LAYERS = 12;
else
    if (modelName == "gpt2-large" || modelName == "gpt2-large-mod")
    NUM_LAYERS = 48;
    else
    end
end

clear files
%% Run GPT

% define generation steps
Steps = 20;

% input Tokens start with empty array
inputTokens = [];
% initial input is loaded into the 'out' array for identical loop structure
out = [2061,  318,  534, 1438,   30];

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

%% Print output
final_out = [inputTokens out]
