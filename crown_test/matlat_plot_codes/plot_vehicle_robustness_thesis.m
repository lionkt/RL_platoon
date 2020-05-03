test_vehicle_ratio = 0.8:0.2:2;
acc_speed_rmse = [1.506,1.46,1.45,1.45,1.451,1.453]; %0.8下碰撞
acc_dist_error = [0.81,0.758,0.76,0.76,0.76,0.759]; %0.8下碰撞
cacc_speed_rmse = [0.69,0.604,0.59,0.59,0.59,0.594,0.596]; 
cacc_dist_error = [1.079,0.349,0.331,0.334,0.333,0.333,0.333];
cacc_into_bound_time = [64.8, 64.35, 65, 64.9, 64.9,65]; %0.8没进入误差带
rl_speed_rmse = [0.594,0.37,0.288,0.24,0.207,0.182,0.163];
rl_dist_error=[0.34,0.266,0.232,0.211,0.198,0.188,0.18];
rl_into_bound_time=[63.95,63.45,63.2,62.4,62.5,61.9,62.4];

%% plot
figure();
plot(test_vehicle_ratio(2:end), acc_speed_rmse,'--*','linewidth',1.5); hold on;
plot(test_vehicle_ratio, cacc_speed_rmse, '--+','linewidth',1.5);
plot(test_vehicle_ratio, rl_speed_rmse, '-o','linewidth',1.5);
% plot(test_dist(1), rl_time_bound(1),'bo','markersize',7);
% plot(test_dist(1), cacc_time_bound(1),'r*','markersize',9);
ylim([0.1, 1.6]);
legend('CTG-ACC','滑模CACC','连续奖励型DDPG');
grid on;
xlabel('参数p_{veh}取值');
ylabel('跟驰车与领航车速度RMSE (m/s)');

%% plot 
figure();
plot(test_vehicle_ratio(2:end), acc_dist_error,'--*','linewidth',1.5); hold on;
plot(test_vehicle_ratio, cacc_dist_error, '--+','linewidth',1.5);
plot(test_vehicle_ratio, rl_dist_error, '-o','linewidth',1.5);
% plot(test_dist(1), rl_time_bound(1),'bo','markersize',7);
% plot(test_dist(1), cacc_time_bound(1),'r*','markersize',9);
ylim([0.1, 1.2]);
legend('CTG-ACC','滑模CACC','连续奖励型DDPG');
grid on;
xlabel('参数p_{veh}取值');
ylabel('车队间距误差\epsilon (m)');

%% plot
figure();
plot(test_vehicle_ratio(2:end), cacc_into_bound_time, '--+','linewidth',1.5); hold on;
plot(test_vehicle_ratio, rl_into_bound_time, '-o','linewidth',1.5);
% plot(test_dist(1), rl_time_bound(1),'bo','markersize',7);
% plot(test_dist(1), cacc_time_bound(1),'r*','markersize',9);
% ylim([0.1, 1.2]);
legend('滑模CACC','连续奖励型DDPG');
grid on;
xlabel('参数p_{veh}取值');
ylabel('车队进入5%误差带平均用时 (s)');
