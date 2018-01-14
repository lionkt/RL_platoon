function [ action, score ] = get_action_score( env_param, state, theta )
%%% calculate score and action according to now state

y = state.y;    % centralized state
phi_x = zeros(env_param.state_feature_num,1);   % kernel for RBN
mu = zeros(env_param.act_num,1);    % probability for action

for th_ = 1:env_param.state_feature_num
    err = y - env_param.grid_center(:,th_);
    phi_x(th_) = exp(-0.5 * err' * env_param.inv_sig_grid_network * err);
end

%% calculate action probability
for th_ = 1:env_param.act_num
    if (th_ == 1)
        phi_xa = [phi_x; zeros(env_param.state_feature_num,1);];    
    else
        phi_xa = [zeros(env_param.state_feature_num,1); phi_x];
    end
    mu(th_) = exp(phi_xa' * theta);
end
mu = mu / sum(mu);  % normalized
%%% sample from probability
sample_val = rand;
if (sample_val < mu(1))
    action = env_param.action(1);
    score = [phi_x * (1 - mu(1)); -phi_x * mu(2)];
else
    action = env_param.action(2);
    score = [-phi_x * mu(1); phi_x * (1 - mu(2))];
end



end

