% podatki
data = [
5	13.94	0.06	61.32	1.89
10	24.37	0.03	85.81	0.38
15	35.60	0.05	88.46	0.08
20	48.65	0.21	89.21	0.04
25	61.30	0.24	89.48	0.03
30	73.56	0.45	89.67	0.04
35	84.66	0.21	89.81	0.04
40	96.95	0.18	89.80	0.04
45	108.69	0.16	89.89	0.04
50	121.19	0.43	89.98	0.04
];

N = data(:,1);
time = data(:,2);
time_se = data(:,3);
ca = data(:,4);
ca_se = data(:,5);

% aproksimacija s crto
coeffs = polyfit(N,time,1);
fit_N = [min(N), max(N)];
fit_time = polyval(coeffs, fit_N);

% narisi cas
figure(1); clf;
hold on;
plot(fit_N, fit_time, ':', 'LineWidth', 1.5);
plot(N, time, '.', 'markersize', 20);
hold off;
grid on;
xlabel('N');
ylabel('Èas [s]');

% aproksimacija exp
f = fit(N, ca, fittype('exp2'));

% narisi CA
figure(2); clf;
hold on;
x = linspace(min(N), max(N), 200);
plot(x, f(x), ':', 'LineWidth', 1.5);
plot(N, ca, '.', 'markersize', 20);
hold off;
grid on;
xlabel('N');
ylabel('klasifikacijska toènost [%]');