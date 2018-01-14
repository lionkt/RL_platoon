function [ env_param ] = create_env_param(  )
%%% build environment parameters
%%% mountain car environment for Reinforcemnt learning task
%%% speed: -0.07 ~ 0.07, location: -1.2 ~ 0.5, goal: 0.5
env_param.pos_range = [-1.2; 0.5];
env_param.vel_range = [-0.07; 0.07];
env_param.goal = env_param.pos_range(2);
env_param.pos_map_range = [0;1];
env_param.vel_map_range = [0;1];
env_param.grid_size = [4;4];

%% create grid for RB networks
% A 4*4 grid networks for mapping observation state to action space
env_param.c_map_pos = [env_param.pos_range(1), 1; env_param.pos_range(2), 1] \...
    [env_param.pos_map_range(1); env_param.pos_map_range(2)];
env_param.c_map_vel = [env_param.vel_range(1), 1; env_param.vel_range(2), 1] \...
    [env_param.vel_map_range(1); env_param.vel_map_range(2)];
env_param.grid_step = [(env_param.pos_map_range(2) - env_param.pos_map_range(1)) / env_param.grid_size(1);  % gridµÄcenter¼ä¸ô
    (env_param.vel_map_range(2) - env_param.vel_map_range(1)) / env_param.grid_size(2)];
env_param.state_feature_num = env_param.grid_size(1) * env_param.grid_size(2);
% calculate grid network centers
env_param.grid_center = zeros(2,env_param.state_feature_num);   
for i = 1:env_param.grid_size(1)
    for j = 1:env_param.grid_size(2)
        env_param.grid_center(:,((i - 1) * env_param.grid_size(2)) + j) = ...
            [env_param.pos_map_range(1) + ((i - 0.5) * env_param.grid_step(1)); ...
            env_param.vel_map_range(1) + ((j - 0.5) * env_param.grid_step(2))];
    end
end

%% calculate sigma for RB networks
env_param.sig_grid = 1.3 * env_param.grid_step(1);
env_param.sig_grid2 = env_param.sig_grid^2;
env_param.sig_grid_network = env_param.sig_grid2 * eye(2);
env_param.inv_sig_grid_network = inv(env_param.sig_grid_network);

env_param.act_num = 2;  % action space dim
env_param.act = [-1; 1];    % action
env_param.num_policy_param = env_param.state_feature_num * env_param.act_num;


end

