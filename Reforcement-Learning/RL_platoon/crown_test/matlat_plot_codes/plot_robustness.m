clc;
clear;

ACC_loca_data = [84.9;84.1;83.3;82.9;82.5;82.1];
ACC_speed_data = [0.348; 0.352; 0.358; 0.359; 0.355; 0.359];

RL_loca_data = [50.8; 50.8; 50.7; 50.7; 50.7; 50.7];
RL_speed_data = [0.312; 0.245; 0.199; 0.174; 0.158; 0.148];
p = [0.85;1; 1.25; 1.5; 1.75; 2];


figure;
xx1 = plot(p, ACC_loca_data, 'linewidth',1.5); hold on;
xx2 = plot(p, RL_loca_data, 'linewidth',1.5); hold on;
xx3 = plot(p, ACC_loca_data, 'ob','linewidth',1.5); hold on;
xx4 = plot(p, RL_loca_data,'*r', 'linewidth',1.5); hold on;
xlabel('Proportion to Acceleration (p)');
ylabel('within 5% spcing band time (s)');
title('Effect of p on Controller Convergence Speed','Fontweight','bold');
legend([xx3,xx4],'ACC controller','DDPG controller');
ylim([40,85]);
grid on;

figure;
xx1 = plot(p, ACC_speed_data, 'linewidth',1.5); hold on;
xx2 = plot(p, RL_speed_data, 'linewidth',1.5); hold on;
xx3 = plot(p, ACC_speed_data, 'ob','linewidth',1.5); hold on;
xx4 = plot(p, RL_speed_data,'*r', 'linewidth',1.5); hold on;
xlabel('Proportion to Acceleration (p)');
ylabel('velocity RMSE (m/s)');
title('Effect of p on Velocity RMSE','Fontweight','bold');
legend([xx3,xx4],'ACC controller','DDPG controller');
grid on;
