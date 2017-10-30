clc;
clear all;

path = './OutputImg/';
speed_path = [path, 'speed_data.txt'];
acc_path = [path, 'acc_data.txt'];
jerk_path = [path, 'jerk_data.txt'];
inter_dist_path = [path, 'inter-distance_data.txt'];
speed_data = importdata(speed_path);
acc_data = importdata(acc_path);
jerk_data = importdata(jerk_path);
inter_dist_data = importdata(inter_dist_path);
Time_step = 0.2;
Car_length = 5.0;


figure;
suptitle('dynamics');
time1 = [0:1:size(speed_data,1)-1]*Time_step;
subplot(311);
for i=1:size(speed_data,2)
    plot(time1,speed_data,'linewidth',1.5);
    hold on;
end
grid on;
subplot(312);
time = [0:1:size(acc_data,1)-1]*Time_step;
for i=1:size(acc_data,2)
    plot(time,acc_data,'linewidth',1.5);
    hold on;
end
grid on;
subplot(313);
time3 = [0:1:size(jerk_data,1)-1]*Time_step;
for i=1:size(jerk_data,2)
    plot(time3,jerk_data,'linewidth',1.5);
    hold on;
end
grid on;


figure;
suptitle('location');
subplot(211);
time = [0:1:size(inter_dist_data,1)-1]*Time_step;
for i=1:size(inter_dist_data,2)
    plot(time,inter_dist_data,'linewidth',1.5);
    hold on;
end
grid on;
subplot(212);
correction = 0.0;   % 或者去掉车的长度Car_length
for i=1:size(inter_dist_data,2)-1
    inter_dist_1 = inter_dist_data(:,i);
    inter_dist_2 = inter_dist_data(:,i+1);
    yf_norm = abs(fft(inter_dist_2 - correction) ./ fft(inter_dist_1 - correction));
    yf_norm_falf = yf_norm(1:int32(length(yf_norm)/2));   
    plot(yf_norm_falf,'linewidth',1.5);
    hold on;
end
grid on;
    
    
    


