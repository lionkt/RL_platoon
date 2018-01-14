clc;
clear;
output_path = './train_output/MCPG/';
if ~exist(output_path,'dir')
    mkdir(output_path);
end
%% init learning parameters
s = RandStream('mt19937ar','Seed',2);   % 设置随机数的种子
RandStream.setGlobalStream(s);

learning_param.max_episode_length = 200;
learning_param.max_update_num = 500;
learning_param.eval_interval = 50;
learning_param.eval_episode_num = 100;
% learning_param.gamma = 0.95;
learning_param.train_episode_num = 10;


%% init environment and traing parameters
env_param = create_env_param();







%% begin calculation





%% plot output






