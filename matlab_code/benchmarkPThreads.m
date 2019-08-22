clear; clc;

% podatki
data = [
25.766 47.139 67.710 89.321 111.376
12.357 24.162 36.318 47.401  60.982
 6.786 20.326 27.343 37.005  47.910
 4.802 12.928 20.191 26.228  33.213
 6.212 19.525 20.234 32.431  42.443
];

N = [5000, 10000, 15000, 20000, 25000]; % velikosti problema
T = [2, 4, 8, 16]; % stevilo niti

% cas
figure(1); clf;
hold on;
plot(N, data(1,:), '.', 'markersize', 20);
plot(N, data(2,:), '.', 'markersize', 20);
plot(N, data(3,:), '.', 'markersize', 20);
plot(N, data(4,:), '.', 'markersize', 20);
plot(N, data(5,:), '.', 'markersize', 20);
legend('1 nit', '2 niti', '4 niti', '8 niti', '16 niti', 'Location','northwest');
ax = gca;
ax.ColorOrderIndex = 1;
plot(N, data(1,:), '--', 'LineWidth', 1.5);
plot(N, data(2,:), '--', 'LineWidth', 1.5);
plot(N, data(3,:), '--', 'LineWidth', 1.5);
plot(N, data(4,:), '--', 'LineWidth', 1.5);
plot(N, data(5,:), '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Èasovne meritve');
xlabel('Velikost uène množice [N]');
ylabel('Èas [s]');

% pohitritev v odvisnosti od stevila niti
S = data(2:end,:);
S(1,:) = data(1,:) ./ S(1,:);
S(2,:) = data(1,:) ./ S(2,:);
S(3,:) = data(1,:) ./ S(3,:);
S(4,:) = data(1,:) ./ S(4,:);

figure(2); clf;
hold on;
plot(T, S(:,1), '.', 'markersize', 20);
plot(T, S(:,2), '.', 'markersize', 20);
plot(T, S(:,3), '.', 'markersize', 20);
plot(T, S(:,4), '.', 'markersize', 20);
plot(T, S(:,5), '.', 'markersize', 20);
legend('N = 5000', 'N = 10000', 'N = 15000', 'N = 20000', 'N = 25000', 'Location','northwest');
ax = gca;
ax.ColorOrderIndex = 1;
plot(T, S(:,1), '--', 'LineWidth', 1.5);
plot(T, S(:,2), '--', 'LineWidth', 1.5);
plot(T, S(:,3), '--', 'LineWidth', 1.5);
plot(T, S(:,4), '--', 'LineWidth', 1.5);
plot(T, S(:,5), '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Pohitritev');
xlabel('Število niti [T]');
ylabel('Pohitritev');

% ucinkovitost v odvisnosti od stevila niti
E = S;
E(1,:) = E(1,:) ./ T(1);
E(2,:) = E(2,:) ./ T(2);
E(3,:) = E(3,:) ./ T(3);
E(4,:) = E(4,:) ./ T(4);

figure(3); clf;
hold on;
plot(T, E(:,1), '.', 'markersize', 20);
plot(T, E(:,2), '.', 'markersize', 20);
plot(T, E(:,3), '.', 'markersize', 20);
plot(T, E(:,4), '.', 'markersize', 20);
plot(T, E(:,5), '.', 'markersize', 20);
legend('N = 5000', 'N = 10000', 'N = 15000', 'N = 20000', 'N = 25000', 'Location','southwest');
ax = gca;
ax.ColorOrderIndex = 1;
plot(T, E(:,1), '--', 'LineWidth', 1.5);
plot(T, E(:,2), '--', 'LineWidth', 1.5);
plot(T, E(:,3), '--', 'LineWidth', 1.5);
plot(T, E(:,4), '--', 'LineWidth', 1.5);
plot(T, E(:,5), '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Uèinkovitost');
xlabel('Število niti [T]');
ylabel('Uèinkovitost');