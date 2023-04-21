function plot_singular_dist(singular_values)

[r, col] = size(singular_values);

figure
hold on
for i = 1:r
    [idx, c] = singular_dist(singular_values(i,:));
    plot(idx/col * 100, c * 100);
end

plot(idx/col * 100, idx/col * 100, 'LineStyle','--');
xlabel("Percentage of singular values in each head (%)");
ylabel("Percentage of the sum of all singular values (%)");
xlim([1/col * 100, 100])
ylim([1/col * 100, 100])
grid on
end