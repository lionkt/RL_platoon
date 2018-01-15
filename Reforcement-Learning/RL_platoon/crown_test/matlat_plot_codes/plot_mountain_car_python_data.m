clc;
clear;
path = '../OutputImg/Mountain_car/';
DQN_avg_step_name_list = dir([path,'DQN*']);
AC_avg_step_name_list = dir([path,'AC*']);
DDPG_avg_step_name_list = dir([path,'DDPG*']);
exist_flag = zeros(3,1);
if length(DQN_avg_step_name_list) >=1
    DQN_avg_step = importdata([path,DQN_avg_step_name_list(1).name]);
    exist_flag(1) = 1;
end
if length(AC_avg_step_name_list) >=1
    AC_avg_step = importdata([path,AC_avg_step_name_list(1).name]);
    exist_flag(2) = 1;
end
if length(DDPG_avg_step_name_list) >=1
    DDPG_avg_step = importdata([path,DDPG_avg_step_name_list(1).name]);
    exist_flag(3) = 1;
end

sample_interval = 50;
max_train_eps = 500;

%% plot
str_ = [];
figure;
for th_ = 1:length(exist_flag)
    if th_==1 && exist_flag(th_)~=0
        str_ = [str_, '''DQN'','];
        hd = plot(DQN_avg_step,'linewidth',1.5);
        hold on;
    elseif th_==2 && exist_flag(th_)~=0
        str_ = [str_, '''Actor-Critic'','];
        hd = plot(AC_avg_step,'linewidth',1.5);
        hold on;
    elseif th_==3 && exist_flag(th_)~=0
        str_ = [str_, '''DDPG'''];
        hd = plot(DDPG_avg_step,'linewidth',1.5);
        hold on;
    end
    
end
legend( str_);
grid on; xlabel('iter-times');ylabel('mean-steps');






