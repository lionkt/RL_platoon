clc;
clear;
path = '../Data_thesis/';
detailed_path = [path, '5car/ACC/'];
% detailed_path = [path, '5car/CACC/'];
% detailed_path = [path, '5car/RL/continuous/'];       %从第2个follower开始全用RL
% detailed_path = [path, '5car/RL/continuous_allRL/']; %从第1个follower开始全用RL
speed_path = [detailed_path,'speed_data.txt'];
acc_path = [detailed_path, 'acc_data.txt'];
jerk_path = [detailed_path, 'jerk_data.txt'];
location_path = [detailed_path, 'location_data.txt'];
inter_dist_path = [detailed_path, 'inter-distance_data.txt'];
speed_data = importdata(speed_path);
acc_data = importdata(acc_path);
jerk_data = importdata(jerk_path);
location_data = importdata(location_path);
inter_dist_data = importdata(inter_dist_path);
PLOT_FLAG = false;  % ----------- 是否画图 -------------

%% 道路相关
Time_Step = 0.2;
Car_Length = 5;
Desired_inter_distance = 5;
MAX_V = 60/3.6; %m/s
TIME_TAG_UP_BOUND = 120;
ROAD_LENGTH = MAX_V * TIME_TAG_UP_BOUND;
START_LEADER_TEST_DISTANCE = ROAD_LENGTH / 1.4;
inter_dist_data = inter_dist_data + Car_Length;

%% 计算数值指标
disp(['START_LEADER_TEST_DISTANCE:',num2str(START_LEADER_TEST_DISTANCE)]);
disp('开始测试后的误差');
disp('======location误差为:');
sumup =  0;
ixx = find(location_data(:,1)>=START_LEADER_TEST_DISTANCE);
ixx1 = ixx;
if max(ixx1)>size(inter_dist_data,1)
    ixx1(ixx1>size(inter_dist_data,1)) = [];
end
for i=1:size(inter_dist_data,2)
    temp_err = inter_dist_data(ixx1,i) - ones(length(inter_dist_data(ixx1,i)),1)*(Desired_inter_distance+Car_Length);
    temp_rmse = sqrt(mean(temp_err.^2));
    if i>=1
        sumup = sumup+temp_rmse^2;
    end
    disp(['第',num2str(i),'F车和前车的location RSME为：',num2str(temp_rmse),'m']);
end
disp(['从第1辆F车到第',num2str(size(inter_dist_data,2)),'的location RMSE的均值为:',num2str(sqrt(sumup/(size(inter_dist_data,2))))]);

disp('======第一次接近desired spcing的时刻为:');
sumup = 0;
percent = 5e-2;
for i=1:size(inter_dist_data,2)
   my_inter_dist = inter_dist_data(:,i);
   time1 = [1:1:length(my_inter_dist)]*Time_Step;
   index = find(abs(my_inter_dist - (Desired_inter_distance+Car_Length)) < percent*(Desired_inter_distance+Car_Length));
   disp(['第',num2str(i),'F车首次进入desired-dist的时刻为：',num2str(time1(index(1))),'s']);
   if i>=1
       sumup = sumup + time1(index(1));
   end
end
disp(['从第2辆F车到第',num2str(size(inter_dist_data,2)),'的时刻均值为:',num2str(sumup/(size(inter_dist_data,2))),'s']);

disp('======速度误差为:');
sumup =  0;
ixx2 = ixx;
if max(ixx2)>size(speed_data,1)
    ixx2(ixx2>size(speed_data,1)) = [];
end
leader_test_speed = speed_data(ixx2,1);
for i=2:size(location_data,2)
    temp_err = speed_data(ixx2,i)-leader_test_speed;
    temp_rmse = sqrt(mean(temp_err.^2));
    if i>1
        sumup = sumup+temp_rmse^2;
    end
    disp(['第',num2str(i-1),'F车和前车的speed RMSE为：',num2str(temp_rmse),'m/s']);
end
disp(['从第1辆F车到第',num2str(size(inter_dist_data,2)),'的speed RMSE的均值为:',num2str(sqrt(sumup/(size(inter_dist_data,2))))]);

%% plot dynamics
if PLOT_FLAG
    figure;
    % suptitle('dynamics');
    subplot(211);
    time1 = [1:1:size(speed_data,1)]*Time_Step;
    max_speed_arr = ones(size(speed_data,1),1)*MAX_V;
    plot(time1,max_speed_arr,'--r','linewidth',1.7);
    hold on;
    for i=1:size(speed_data,2)
        if i==1
            plot(time1, speed_data(:,i),':','linewidth',1.5,'color',[0.93,0.69,0.13]);
        elseif i==2
            plot(time1, speed_data(:,i),'-.','linewidth',1.5,'color',[0.47,0.67,0.19]);
        elseif i==3
            plot(time1, speed_data(:,i),'linewidth',1.3,'color',[0,0.45,0.74]);
        elseif i==4
            plot(time1, speed_data(:,i),'linewidth',1.3,'color',[0.85,0.33,0.1]);
        elseif i==5
            plot(time1, speed_data(:,i),'linewidth',1.3,'color',[0.49,0.18,0.56]);
        else
            plot(time1, speed_data(:,i),'linewidth',1.3);
        end
        hold on;
    end
    title('velocity');
    ylabel('m/s');xlabel('time stamp(s)');
    % set(gca,'xticklabel',[]);    %隐藏x轴
    grid on;

    subplot(212);
    time2 = [1:1:size(acc_data,1)]*Time_Step;
    for i=1:size(acc_data,2)
        if i==1
            plot(time1, acc_data(:,i),':','linewidth',1.5,'color',[0.93,0.69,0.13]);
        elseif i==2
            plot(time1, acc_data(:,i),'-.','linewidth',1.5,'color',[0.47,0.67,0.19]);
        elseif i==3
            plot(time1, acc_data(:,i),'linewidth',1.3,'color',[0,0.45,0.74]);
        elseif i==4
            plot(time1, acc_data(:,i),'linewidth',1.3,'color',[0.85,0.33,0.1]);
        elseif i==5
            plot(time1, acc_data(:,i),'linewidth',1.3,'color',[0.49,0.18,0.56]);
        else
            plot(time1, acc_data(:,i),'linewidth',1.3);
        end
        hold on;
    end
    title('acceleration');
    xlabel('time stamp(s)');ylabel('m/s^2');
    grid on;
    % subplot(313);
    % time3 = [1:1:size(jerk_data,1)]*Time_Step;
    % for i=1:size(jerk_data,2)
    %     plot(time3, jerk_data(:,i),'linewidth',1.7);
    %     hold on;
    % end
    % grid on;
end
%% location
if PLOT_FLAG
    figure;
    % suptitle('location');
    subplot(211);
    time1 = [1:1:size(inter_dist_data,1)]*Time_Step;
    des_inter_dist = ones(size(inter_dist_data,1),1)*(Desired_inter_distance+Car_Length);
    up_inter_dist_error = ones(size(inter_dist_data,1),1)*(Desired_inter_distance+Car_Length)*(1+percent);
    down_inter_dist_error = ones(size(inter_dist_data,1),1)*(Desired_inter_distance+Car_Length)*(1-percent);
    plot(time1,des_inter_dist,'--r','linewidth',1.2);
    hold on;
    plot(time1,up_inter_dist_error,'--b','linewidth',1,'color',[0.5,0.5,0.5]);
    hold on;
    plot(time1,down_inter_dist_error,'--b','linewidth',1,'color',[0.5,0.5,0.5]);
    hold on;
    for i=1:size(inter_dist_data,2)
        if i==1
            plot(time1, inter_dist_data(:,i),'--','linewidth',1.5,'color',[0.93,0.69,0.13]); hold on;
        elseif i==2 
            plot(time1, inter_dist_data(:,i),'linewidth',1.7,'color',[0,0.45,0.74]); hold on;
        elseif i==3 
            plot(time1, inter_dist_data(:,i),'linewidth',1.7,'color',[0.85,0.33,0.1]); hold on;
        elseif i==4 
            plot(time1, inter_dist_data(:,i),'linewidth',1.7,'color',[0.49,0.18,0.56]); hold on;
        else
            plot(time1, inter_dist_data(:,i),'linewidth',1.7); hold on;
        end
    end
    title('Inter-vehicle Spacing');
    ylabel('m');xlabel('time stamp (s)');
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
    title('Frequency Domain Error Ratio');
    ylabel('amplitude');xlabel('frequency');
end



