clc;
clear;
BP_path = '../OutputImg/Mountain_car/BP_method/';
RBN_path = '../OutputImg/Mountain_car/RBN_method/';
%%% 加载神经网络得到的mountain-car数据
DQN_avg_step_name_list = dir([BP_path,'DQN*']);
AC_avg_step_name_list = dir([BP_path,'AC*']);
DDPG_avg_step_name_list = dir([BP_path,'DDPG*']);
%%% 加载RBN方法得的mountain-car数据
BAC_name_list = {[RBN_path,'BAC gau1 s1234 alpha 0.03 (0 0 - 500.00)  numepisodes 5 sample 50 max 500/'];
    [RBN_path,'BAC gau1 s1234 alpha 0.03 (0 0 - 500.00)  numepisodes 10 sample 50 max 500/'];
    [RBN_path,'BAC gau1 s1234 alpha 0.03 (0 0 - 500.00)  numepisodes 20 sample 50 max 500/'];
    [RBN_path,'BAC gau1 s1234 alpha 0.03 (0 0 - 500.00)  numepisodes 40 sample 50 max 500/']};
BAC_file_list = cell(length(BAC_name_list),1);
for th_ = 1:length(BAC_name_list)
    temp = dir([BAC_name_list{th_},'*.txt']);
    BAC_file_list{th_} = temp.name;
end
MCPG_name_list = {[RBN_path,'MCPG alpha 0.03 numepisodes 5 sample 50 max 500/'];
    [RBN_path,'MCPG alpha 0.03 numepisodes 10 sample 50 max 500/'];
    [RBN_path,'MCPG alpha 0.03 numepisodes 20 sample 50 max 500/'];
    [RBN_path,'MCPG alpha 0.03 numepisodes 40 sample 50 max 500/']};
MCPG_file_list = cell(length(MCPG_name_list),1);
for th_ = 1:length(MCPG_name_list)
    temp = dir([MCPG_name_list{th_},'*.txt']);
    MCPG_file_list{th_} = temp.name;
end  

%% 开始提取数据 \
%%% 神经网络得到的mountain-car数据
BP_exist_flag = zeros(3,1);
if length(DQN_avg_step_name_list) >=1
    DQN_avg_step = importdata([BP_path,DQN_avg_step_name_list(1).name]);
    BP_exist_flag(1) = 1;
end
if length(AC_avg_step_name_list) >=1
    AC_avg_step = importdata([BP_path,AC_avg_step_name_list(1).name]);
    BP_exist_flag(2) = 1;
end
if length(DDPG_avg_step_name_list) >=1
    DDPG_avg_step = importdata([BP_path,DDPG_avg_step_name_list(1).name]);
    BP_exist_flag(3) = 1;
end
%%% RBN方法得的mountain-car数据
BAC_avg_step = [];
MCPG_avg_step = [];
for th_=1:length(BAC_file_list)
    temp = importdata([BAC_name_list{th_},BAC_file_list{th_}]);
    BAC_avg_step = [BAC_avg_step, temp(:,2)];
end
for th_=1:length(MCPG_file_list)
    temp = importdata([MCPG_name_list{th_},MCPG_file_list{th_}]);
    MCPG_avg_step = [MCPG_avg_step, temp(:,2)];
end

%%% 控制字段
sample_interval = 50;
max_train_eps = 500;
RBN_plot_length = 3;    %控制RBN方法画出图形的个数（只画到更新样本20个的时候）

%% plot
str_ = [];
iter_list = [0:sample_interval:max_train_eps];
figure;
for th_ = 1:length(BP_exist_flag)
    if th_==1 && BP_exist_flag(th_)~=0
        str_ = [str_, '''DQN'','];
        hd = plot(iter_list,DQN_avg_step,'linewidth',1.5);
        hold on;
    elseif th_==2 && BP_exist_flag(th_)~=0
        str_ = [str_, '''Actor-Critic'','];
        hd = plot(iter_list,AC_avg_step,'linewidth',1.5);
        hold on;
    elseif th_==3 && BP_exist_flag(th_)~=0
        str_ = [str_, '''DDPG'''];
        hd = plot(iter_list,DDPG_avg_step,'linewidth',1.5);
        hold on;
    end
    
end
legend( str_);
grid on; xlabel('iter-times');ylabel('mean-steps');


figure;
subplot(211);
for th_=1:RBN_plot_length
    plot(iter_list,BAC_avg_step(:,th_),'linewidth',1.5);
    hold on;
end
grid on; title('BAC method');ylim([0, 300]);ylabel('mean-steps');
subplot(212);
for th_=1:RBN_plot_length
    plot(iter_list,MCPG_avg_step(:,th_),'linewidth',1.5);
    hold on;
end
title('MCPG method');
grid on; xlabel('iter-times');ylabel('mean-steps');
