clear; clc;

% podatki - spreminanje velikosti ucne mnozice
% benchIter= [
%     10.811  21.639  32.447  43.243  54.265
%     2.882   5.776   8.639   11.496  14.368
%     1.518   3.036   4.537   6.065   7.571 
%     1.490   2.970   4.477   6.028   7.574 
%     1.039   2.095   3.098   4.147   5.160 
%     1.431   2.875   4.277   5.734   7.145 
% ];

benchDataset = [
    27.143  54.143  81.064  107.983 135.349
    7.240   14.399  21.476  28.532  35.754 
    3.849   7.567   11.288  14.995  18.690 
    3.872   7.319   10.748  14.575  17.373 
    2.836   5.212   7.576   9.905   12.387 
    4.161   7.127   9.503   12.268  15.144 
];

N = [5000, 10000, 15000, 20000, 25000]'; % velikosti problema
T = [4, 8, 16, 24, 32]'; % stevilo niti

figure(1); clf;
hold on;
plot(N, benchDataset(1,:), '.', 'markersize', 20);
plot(N, benchDataset(2,:), '.', 'markersize', 20);
plot(N, benchDataset(3,:), '.', 'markersize', 20);
plot(N, benchDataset(4,:), '.', 'markersize', 20);
plot(N, benchDataset(5,:), '.', 'markersize', 20);
plot(N, benchDataset(6,:), '.', 'markersize', 20);
legend('1 nit', '4 niti', '8 niti', '16 niti', '24 niti', '32 niti', 'Location','northwest');
ax = gca;
ax.ColorOrderIndex = 1;
plot(N, benchDataset(1,:), '--', 'LineWidth', 1.5);
plot(N, benchDataset(2,:), '--', 'LineWidth', 1.5);
plot(N, benchDataset(3,:), '--', 'LineWidth', 1.5);
plot(N, benchDataset(4,:), '--', 'LineWidth', 1.5);
plot(N, benchDataset(5,:), '--', 'LineWidth', 1.5);
plot(N, benchDataset(6,:), '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Časovne meritve');
xlabel('Velikost učne množice [N]');
ylabel('Čas [s]');

% pohitritev v odvisnosti od stevila niti
S = benchDataset(2:end,:);
S(1,:) = benchDataset(1,:) ./ S(1,:);
S(2,:) = benchDataset(1,:) ./ S(2,:);
S(3,:) = benchDataset(1,:) ./ S(3,:);
S(4,:) = benchDataset(1,:) ./ S(4,:);
S(5,:) = benchDataset(1,:) ./ S(5,:);

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
E(5,:) = E(5,:) ./ T(5);

figure(3); clf;
hold on;
plot(T, E(:,1), '.', 'markersize', 20);
plot(T, E(:,2), '.', 'markersize', 20);
plot(T, E(:,3), '.', 'markersize', 20);
plot(T, E(:,4), '.', 'markersize', 20);
plot(T, E(:,5), '.', 'markersize', 20);
legend('N = 5000', 'N = 10000', 'N = 15000', 'N = 20000', 'N = 25000', 'Location','northeast');
ax = gca;
ax.ColorOrderIndex = 1;
plot(T, E(:,1), '--', 'LineWidth', 1.5);
plot(T, E(:,2), '--', 'LineWidth', 1.5);
plot(T, E(:,3), '--', 'LineWidth', 1.5);
plot(T, E(:,4), '--', 'LineWidth', 1.5);
plot(T, E(:,5), '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Učinkovitost');
xlabel('Število niti [T]');
ylabel('Učinkovitost');
