INIT_CAR_DISTANCE = 25;
test_dist = [0,5,10,15,20,25];
test_ratio = test_dist/INIT_CAR_DISTANCE;
rl_time_bound = [63.45, 65.7,67,68.15,69.05,70.95];
rl_spacing_error= [0.266, 0.28,0.279,0.282,0.285,0.289];
rl_speed_rmse = [0.37,0.37,0.37,0.372,0.367,0.368];
cacc_time_bound = [64.8, 69.05, 70.95, 73.2, 74, 74.75];
cacc_spacing_error = [0.35,0.355,0.364,0.387,0.42,0.458];
cacc_speed_rmse = [0.604,0.6,0.608,0.611,0.604,0.608];

%% plot
figure();
plot(test_ratio*100, rl_time_bound,'-o','linewidth',1.5); hold on;
plot(test_ratio*100, cacc_time_bound, '--*','linewidth',1.5);
plot(test_dist(1), rl_time_bound(1),'bo','markersize',7);
plot(test_dist(1), cacc_time_bound(1),'r*','markersize',9);
legend('连续奖励型DDPG','滑模CACC');
grid on;
xlabel('d_{init}扩增比例 (%)');
ylabel('跟驰车进入5%误差带平均用时 (s)');

figure();
plot(test_ratio*100, rl_spacing_error,'-o','linewidth',1.5); hold on;
plot(test_ratio*100, cacc_spacing_error, '--*','linewidth',1.5);
plot(test_dist(1), rl_spacing_error(1),'bo','markersize',7);
plot(test_dist(1), cacc_spacing_error(1),'r*','markersize',9);
legend('连续奖励型DDPG','滑模CACC');
grid on;
xlabel('d_{init}扩增比例 (%)');
ylabel('车队间距误差\epsilon (m)');


figure();
plot(test_ratio*100, rl_speed_rmse,'-o','linewidth',1.5); hold on;
plot(test_ratio*100, cacc_speed_rmse, '--*','linewidth',1.5);
plot(test_dist(1), rl_speed_rmse(1),'bo','markersize',7);
plot(test_dist(1), cacc_speed_rmse(1),'r*','markersize',9);
legend('连续奖励型DDPG','滑模CACC');
grid on;
xlabel('d_{init}扩增比例 (%)');
ylabel('跟驰车与领航车速度RMSE (m/s)');
