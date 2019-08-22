clear; clc;

mpi = [
    91.87 184.23 274.73 365.041
    39.86  79.82 120.71 158.338
    39.33  40.99  61.29  89.381
    17.60  27.99  26.99  39.338
    12.36  18.48  41.07  40.857
];

N = [10000, 20000, 30000, 40000]; % velikosti problema
P = [10, 30, 40, 50]; % stevilo procesov

% cas
figure(1); clf;
hold on;
plot(N, mpi(1,:), '.', 'markersize', 20);
plot(N, mpi(2,:), '.', 'markersize', 20);
plot(N, mpi(3,:), '.', 'markersize', 20);
plot(N, mpi(4,:), '.', 'markersize', 20);
plot(N, mpi(5,:), '.', 'markersize', 20);
legend('2 procesa', '10 procesov', '30 procesov', '40 procesov', '50 procesov', 'Location','northwest');
ax = gca;
ax.ColorOrderIndex = 1;
plot(N, mpi(1,:), '--', 'LineWidth', 1.5);
plot(N, mpi(2,:), '--', 'LineWidth', 1.5);
plot(N, mpi(3,:), '--', 'LineWidth', 1.5);
plot(N, mpi(4,:), '--', 'LineWidth', 1.5);
plot(N, mpi(5,:), '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Èasovne meritve');
xlabel('Velikost uène množice [N]');
ylabel('Èas [s]');

% pohitritev (v odvisnosti od stevila procesov)
S = mpi(2:end,:);
S(1,:) = mpi(1,:) ./ S(1,:);
S(2,:) = mpi(1,:) ./ S(2,:);
S(3,:) = mpi(1,:) ./ S(3,:);
S(4,:) = mpi(1,:) ./ S(4,:);

figure(2); clf;
hold on;
plot(P, S(:,1), '.', 'markersize', 20);
plot(P, S(:,2), '.', 'markersize', 20);
plot(P, S(:,3), '.', 'markersize', 20);
plot(P, S(:,4), '.', 'markersize', 20);
legend('N = 10000', 'N = 20000', 'N = 30000', 'N = 40000', 'Location','northwest');
ax = gca;
ax.ColorOrderIndex = 1;
plot(P, S(:,1), '--', 'LineWidth', 1.5);
plot(P, S(:,2), '--', 'LineWidth', 1.5);
plot(P, S(:,3), '--', 'LineWidth', 1.5);
plot(P, S(:,4), '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Pohitritev');
xlabel('Število procesov [P]');
ylabel('Pohitritev');

% ucinkovitost (v odvisnosti od stevila procesov)
E = S;
E(1,:) = E(1,:) ./ P(1);
E(2,:) = E(2,:) ./ P(2);
E(3,:) = E(3,:) ./ P(3);
E(4,:) = E(4,:) ./ P(4);

figure(3); clf;
hold on;
plot(P, E(:,1), '.', 'markersize', 20);
plot(P, E(:,2), '.', 'markersize', 20);
plot(P, E(:,3), '.', 'markersize', 20);
plot(P, E(:,4), '.', 'markersize', 20);
legend('N = 10000', 'N = 20000', 'N = 30000', 'N = 40000', 'Location','southwest');
ax = gca;
ax.ColorOrderIndex = 1;
plot(P, E(:,1), '--', 'LineWidth', 1.5);
plot(P, E(:,2), '--', 'LineWidth', 1.5);
plot(P, E(:,3), '--', 'LineWidth', 1.5);
plot(P, E(:,4), '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Uèinkovitost');
xlabel('Število procesov [P]');
ylabel('Uèinkovitost');
