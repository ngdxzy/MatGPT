function plot_singular_dist(singular_values)

[r, ~] = size(singular_values);

figure
hold on
for i = 1:r
    [idx, c] = singular_dist(singular_values(i,:));
    plot(idx, c);
end

end