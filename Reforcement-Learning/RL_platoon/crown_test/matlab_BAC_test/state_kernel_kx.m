function [ kx ] = state_kernel_kx( state, statedic, env_param )
% calculate state kernel 
sigk_x = 1;
ck_x = 1;
x = state.x';
xdic = vertcat(statedic.x)';
y = [env_param.c_map_pos(1); env_param.c_map_vel(1)] .* x;
ydic = repmat([env_param.c_map_pos(1); env_param.c_map_vel(1)],1,size(xdic,2)) .* xdic ;
temp = pdist2(y',ydic').^2;
kx = ck_x * exp(-temp / (2 * sigk_x*sigk_x));


end

