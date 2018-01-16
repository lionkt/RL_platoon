clc;
clear;
output_path = './train_output/BAC/';
if ~exist(output_path,'dir')
    mkdir(output_path);
end

%% init parameters
s = RandStream('mt19937ar','Seed',2);   % 设置随机数的种子
RandStream.setGlobalStream(s);

learning_param.max_episode_length = 200;
learning_param.max_update_num = 500;
learning_param.eval_interval = 50;
learning_param.eval_episode_num = 100;
learning_param.alpha_init = 0.025;
learning_param.train_episode_num = 10;

learning_param.gamma = 0.99;
learning_param.alp_variance_adaptive = 0;
learning_param.alp_schedule = 0;
learning_param.alp_update_param = 500;
learning_param.SIGMA_INIT=1;
%%% init environment and traing parameters
env_param = create_env_param();


%% begin calculation
[ mean_step_list, calc_time_list, theta  ] = BAC( learning_param, env_param );



%% plot output
figure;
plot(mean_step_list,'linewidth',1.5)
title('mean step in Mountain-Car');
xlabel('iter times');











