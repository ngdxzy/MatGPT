function [s] = bpe_decoding(a)

s = sprintf(".%d", a);
s = eraseBetween(s,1,1);

[~, s] = system("./bpe_decoding.py """ + s + """");

end