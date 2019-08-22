clear; clc;

% casovne meritve
serial = [20.389  39.527  62.887  80.625  100.134];
pthreads = [5.475   11.841  19.008  26.357  33.777];
omp = [5.751   13.029  19.956  26.987  34.097];

N = [5000, 10000, 15000, 20000, 25000]'; % velikosti problema
T = 8; % stevilo niti

figure(1); clf;
hold on;
ax = gca;

plot(N, pthreads, '.', 'markersize', 20);
plot(N, omp, '.', 'markersize', 20);
ax.ColorOrderIndex = 5;
plot(N, serial, '.', 'markersize', 20);
legend('Pthreads', 'OpenMP', 'Serial', 'Location','northwest');
ax.ColorOrderIndex = 1;
plot(N, pthreads, '--', 'LineWidth', 1.5);
plot(N, omp, '--', 'LineWidth', 1.5);
ax.ColorOrderIndex = 5;
plot(N, serial, '--', 'LineWidth', 1.5);

hold off;
grid on;
title('Časovne meritve');
xlabel('Velikost učne množice [N]');
ylabel('Čas [s]');

% pohitritev
S_pthreads = serial ./ pthreads;
S_omp = serial ./ omp;

figure(2); clf;
hold on;
ax = gca;

plot(N, S_pthreads, '.', 'markersize', 20);
plot(N, S_omp, '.', 'markersize', 20);
legend('Pthreads', 'OpenMP', 'Location','northeast');
ax.ColorOrderIndex = 1;
plot(N, S_pthreads, '--', 'LineWidth', 1.5);
plot(N, S_omp, '--', 'LineWidth', 1.5);

hold off;
grid on;
title('Pohitritev');
xlabel('Velikost učne množice [T]');
ylabel('Pohitritev');
