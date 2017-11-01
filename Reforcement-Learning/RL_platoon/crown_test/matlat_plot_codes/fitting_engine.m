n = [1000:0.01:4000];
Tn = 160.39-0.110913*n + 1.36485*(1e-4)*n.*n -6.191286*(1e-8)*n.^3 + 1.20898*(1e-11)*n.^4 - 8.85607*(1e-16)*n.^5;
T2 = -1.09e-05*n.^2 + 0.05079*n +95.25;

figure;
plot(n, Tn,'linewidth',2);
hold on;
plot(n, T2,'linewidth',2);
grid on;
xlabel('engine speed (r/min)');
ylabel('engine torque (N*m)');
title('Engine Speed-Torque Function of 492Q Engine');

figure;
plot(n ,abs(T2-Tn)./Tn * 100,'linewidth',2);
grid on;
xlabel('engine speed (r/min)');
ylabel('Error amplitude (%)');
title('Error Between Quadratic Fitting and Fifth-order Fitting');
