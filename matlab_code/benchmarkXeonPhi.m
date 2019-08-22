benchmarks = [154.80 10.26 6.66 6.23 7.32 9.56 17.00];

T = [1, 20, 40, 60, 80, 120, 236]; % stevilo niti

% casovne meritve
figure(1); clf;
hold on;
plot(T, benchmarks(1,:), '.', 'markersize', 20);
ax = gca;
ax.ColorOrderIndex = 1;
plot(T, benchmarks(1,:), '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Èasovne meritve');
xlabel('Število niti [T]');
ylabel('Èas [s]');

% pohitritev
serial = 53.615;
S = serial ./ benchmarks;

figure(2); clf;
hold on;
plot(T, S, '.', 'markersize', 20);
ax = gca;
ax.ColorOrderIndex = 1;
plot(T, S, '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Pohitritev');
xlabel('Število niti [T]');
ylabel('Pohitritev');
