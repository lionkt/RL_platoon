function [ n_state ] = mountain_car_step_next( state, action, env_param )
%%% mountain car environment update for Reinforcemnt learning task
%%% speed: -0.07 ~ 0.07, location: -1.2 ~ 0.5, goal: 0.5
%%% speed-transition:x_old + 0.001*acceleration-0.0025*cos(3*location_old)
x_old = state.x;
tmp_vel = x_old(2) + 0.001 * action - 0.0025 * cos(3 * x_old(1));
x(2) = max(env_param.vel_range(1) , min(tmp_vel , env_param.vel_range(2)));
tmp_pos = x_old(1) + x(2);
x(1) = max(env_param.pos_range(1) , min(tmp_pos , env_param.pos_range(2)));

if (x(1) <= env_param.pos_range(1))
    x(2) = 0;
end

if (x(1) >= env_param.goal)
    x(1) = env_param.goal;
    x(2) = 0;
end                 

y = [env_param.c_map_pos(1) * x(1) + env_param.c_map_pos(2);
    env_param.c_map_vel(1) * x(2) + env_param.c_map_vel(2)];

n_state.x = x;
n_state.y = y;
n_state.isgoal = 0;


end

