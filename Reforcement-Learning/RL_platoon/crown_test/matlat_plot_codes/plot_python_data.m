clc;
clear;
path = '../OutputImg/';
speed_path = [path,'speed_data.txt'];
acc_path = [path, 'acc_data.txt'];
jerk_path = [path, 'jerk_data.txt'];
location_path = [path, 'location_data.txt'];
inter_dist_path = [path, 'inter-distance_data.txt'];
speed_data = importdata(speed_path);
acc_data = importdata(acc_path);
jerk_data = importdata(jerk_path);
location_data = importdata(location_path);
inter_dist_data = importdata(inter_dist_path);
%% 道路相关
Time_Step = 0.2;
Car_Length = 5;
Desired_inter_distance = 5;
MAX_V = 60/3.6; %m/s
ROAD_LENGTH = MAX_V * 70;
START_LEADER_TEST_DISTANCE = ROAD_LENGTH / 2;


%% plot data

disp('开始测试后的误差');
disp('location误差为:');
ixx = find(location_data(:,1)>=START_LEADER_TEST_DISTANCE);
ixx1 = ixx;
if max(ixx1)>size(inter_dist_data,1)
    ixx1(ixx1>size(inter_dist_data,1)) = [];
end
for i=1:size(inter_dist_data,2);
    temp_err = inter_dist_data(ixx1,i) - ones(length(inter_dist_data(ixx1,i)),1)*(Desired_inter_distance);
    temp_rmse = sqrt(mean(temp_err.^2));
    disp(['第',num2str(i),'车和前车的location RSME为：',num2str(temp_rmse),'m']);
end

disp('速度误差为:');
ixx2 = ixx;
if max(ixx2)>size(speed_data,1)
    ixx2(ixx2>size(speed_data,1)) = [];
end
leader_test_speed = speed_data(ixx2,1);
for i=2:size(location_data,2);
    temp_err = speed_data(ixx2,i)-leader_test_speed;
    temp_rmse = sqrt(mean(temp_err.^2));
    disp(['第',num2str(i),'车和前车的speed RSME为：',num2str(temp_rmse),'m/s']);
end



figure;
suptitle('dynamics');
subplot(311);
time1 = [1:1:size(speed_data,1)]*Time_Step;
for i=1:size(speed_data,2)
    plot(time1, speed_data(:,i),'linewidth',1.7);
    hold on;
end
grid on;
subplot(312);
time2 = [1:1:size(acc_data,1)]*Time_Step;
for i=1:size(acc_data,2)
    plot(time1, acc_data(:,i),'linewidth',1.7);
    hold on;
end
grid on;
subplot(313);
time3 = [1:1:size(jerk_data,1)]*Time_Step;
for i=1:size(jerk_data,2)
    plot(time3, jerk_data(:,i),'linewidth',1.7);
    hold on;
end
grid on;


figure;
suptitle('location');
subplot(211);
time1 = [1:1:size(inter_dist_data,1)]*Time_Step;
for i=1:size(inter_dist_data,2)
    plot(time1, inter_dist_data(:,i),'linewidth',1.7);
    hold on;
end
grid on;
subplot(212);
for i=1:size(inter_dist_data,2)-1
    inter_dist_1 = inter_dist_data(:,i);
    inter_dist_2 = inter_dist_data(:,i+1);
    yf_norm = abs(fft(inter_dist_2) ./ fft(inter_dist_1));
    yf_norm_half = yf_norm(1:int32(length(yf_norm)/2));
    plot(yf_norm_half,'linewidth',1.7);
    hold on;
end
grid on;




