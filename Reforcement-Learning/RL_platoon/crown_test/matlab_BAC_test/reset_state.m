function [ state ] = reset_state( env_param )
% reset state for training
r1 = rand; r2 = rand;
x = [((env_param.pos_range(2) - env_param.pos_range(1)) * r1) + env_param.pos_range(1);
    ((env_param.vel_range(2) - env_param.vel_range(1)) * r2) + env_param.vel_range(1)];

y = [(env_param.c_map_pos(1) * x(1)) + env_param.c_map_pos(2);
    (env_param.c_map_vel(1) * x(2)) + env_param.c_map_vel(2)];

state.x = x;
state.y = y;     
state.is_goal = 0;

end

