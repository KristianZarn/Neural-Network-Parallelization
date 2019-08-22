clear; clc;

ocl = [
    1.126 1.936 2.750 3.575 4.386 5.192
    1.857 3.422 4.972 6.552 8.083 9.653
    2.609 4.901 7.208 9.505 11.786 14.100
    3.354 6.390 9.424 12.467 15.518 18.552
];

N = [10000, 20000, 30000, 40000, 50000, 60000]; % velikosti problema
I = [50, 100, 150, 200]; % stevilo iteracij

% cas
figure(1); clf;
hold on;
plot(N, ocl(1,:), '.', 'markersize', 20);
plot(N, ocl(2,:), '.', 'markersize', 20);
plot(N, ocl(3,:), '.', 'markersize', 20);
plot(N, ocl(4,:), '.', 'markersize', 20);
legend('50 iteracij', '100 iteracij', '150 iteracij', '200 iteracij', 'Location','northwest');
ax = gca;
ax.ColorOrderIndex = 1;
plot(N, ocl(1,:), '--', 'LineWidth', 1.5);
plot(N, ocl(2,:), '--', 'LineWidth', 1.5);
plot(N, ocl(3,:), '--', 'LineWidth', 1.5);
plot(N, ocl(4,:), '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Èasovne meritve');
xlabel('Velikost uène množice [N]');
ylabel('Èas [s]');

% pohitritev
serial = [
    26.81  53.65  80.41  107.54 134.03 160.81
    53.62  107.16 160.86 217.43 267.69 321.23
    80.41  160.74 241.17 321.55 401.19 482.36
    107.13 213.80 321.63 427.86 535.29 641.85
];
S = serial ./ ocl;

figure(2); clf;
hold on;
plot(N, S(1,:), '.', 'markersize', 20);
plot(N, S(2,:), '.', 'markersize', 20);
plot(N, S(3,:), '.', 'markersize', 20);
plot(N, S(4,:), '.', 'markersize', 20);
legend('50 iteracij', '100 iteracij', '150 iteracij', '200 iteracij', 'Location','southeast');
ax = gca;
ax.ColorOrderIndex = 1;
plot(N, S(1,:), '--', 'LineWidth', 1.5);
plot(N, S(2,:), '--', 'LineWidth', 1.5);
plot(N, S(3,:), '--', 'LineWidth', 1.5);
plot(N, S(4,:), '--', 'LineWidth', 1.5);
hold off;
grid on;
title('Pohitritev');
xlabel('Velikost uène množice [N]');
ylabel('Pohitritev');
