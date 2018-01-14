function [ state ] = reset_state( env_param )
% reset state for training

% x = [((env_param.pos_range(2) - env_param.pos_range(1)) * rand) + env_param.pos_range(1);
%     ((env_param.vel_range(2) - env_param.vel_range(1)) * rand) + env_param.vel_range(1)];

pos_1 = -0.6; pos_2 = -0.4;
vel_1 = env_param.vel_range(1); vel_2 = env_param.vel_range(2);
x = [(pos_2 - pos_1) * rand + pos_1;
    (vel_2 - vel_1) * rand + vel_1];


y = [(env_param.c_map_pos(1) * x(1)) + env_param.c_map_pos(2);
    (env_param.c_map_vel(1) * x(2)) + env_param.c_map_vel(2)];

state.x = x;
state.y = y;     
state.is_goal = 0;

end

