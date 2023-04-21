clear 
fid = fopen("encoder.json","r","n","unicode");
raw = fread(fid,inf);
for i = 1:length(raw)
    if raw(i) > 255
        raw(i) = raw(i) - 256
    end
end
str = char(raw');
fclose(fid)