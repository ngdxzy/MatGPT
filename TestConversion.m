Q = rand(5, 10);
K = rand(5, 10);
x = rand(8, 10);
bq = rand(1,5);
bk = rand(1,5);

[Q1, K1, bq1, bk1] = Convert_Model(Q, K, bq, bk);

y1 = x * Q' + bq;
y2 = x * K' + bk;
z1 = x * Q1' + bq1;
z2 = x * K1' + bk1;
y = y1 * y2';
z = z1 * z2';
sum(sum(y - z))