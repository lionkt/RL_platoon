%单纯的用来绘制quadratic envelope
%%%% 原始的加速度和速度上限
% MAX_ACC = 6;
% MIN_ACC = -10;
MAX_ACC = 2.5;
MIN_ACC = -4;
MAX_V = 60/3.6;

step = 0.001;
speed = [step:step:MAX_V-step];
speed_plot = [0, speed, MAX_V];
% 加速阶段
p = 0.3;
acc_max = MAX_ACC;
v_max = MAX_V;
m = (v_max * p - v_max + v_max * sqrt(1 - p)) / p;
k = (1 - p) * acc_max / m / m;
speed_up_calcValue = -k * (speed - m) .* (speed - m) + acc_max;
speed_up_calcValue = [0, speed_up_calcValue, 0];
% 减速阶段
p = 0.6;
acc_max = MAX_ACC;
v_max = MAX_V;
m = v_max / (sqrt(p) + 1);
k = -MIN_ACC / m / m;
slow_downcalcValue = k * (speed - m) .* (speed - m) + MIN_ACC;
slow_downcalcValue = [0, slow_downcalcValue, 0];

figure;
plot(speed_plot,speed_up_calcValue,'--','linewidth',7);
hold on;
plot(speed_plot,slow_downcalcValue,':','linewidth',7);
fill(speed_plot,speed_up_calcValue,[0.8706 0.9216 0.9804]);
fill(speed_plot,slow_downcalcValue,[0.8706 0.9216 0.9804]);
title('Acc-Vel Quadratic Envelope');
xlabel('velocity (m/s)'); ylabel('acceleraton (m/s^2)');
grid on;


