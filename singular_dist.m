function [idx, c] = singular_dist(singular_value)

m = sum(singular_value);
q = sort(singular_value, 'descend');
q = cumsum(q);
c = q / m;
idx = 1:length(singular_value);


end