function [code] = bpe_encoding(s)

[~, coding] = system("./bpe_encoding.py """ + s + """");
code = eval(coding);

end