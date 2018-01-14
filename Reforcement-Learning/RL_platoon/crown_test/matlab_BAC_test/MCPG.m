function [ mean_step_list, calc_time_list, theta ] = MCPG( learning_param, env_param )
%%% MC policy gradient method for RL problem
% init
alpha = learning_param.alpha_init;
theta = zeros(env_param.policy_param_num,1);
output_num = floor(learning_param.max_update_num / learning_param.eval_interval) + 1;
mean_step_list = zeros(output_num, 1);
calc_time_list = zeros(output_num, 1);

%% begin calculation
tic;
for update_th = 1:learning_param.max_update_num
    if mod(update_th, learning_param.eval_interval)==0
        disp(['MCPG finish ',num2str(update_th/learning_param.max_update_num*100),'% of ',num2str(learning_param.max_update_num)]);
    end
    total_step = 0;
    delta = zeros(env_param.policy_param_num,1);    % update value for theta
    
    %%% begin evaluate
    if update_th==1
        toc;
        eval_ix = 1;
        mean_step_list(eval_ix) = evaluate(learning_param,env_param,theta);        
        tic;
    elseif mod(update_th, learning_param.eval_interval)==0
        toc;
        eval_ix = update_th/learning_param.eval_interval + 1;
        mean_step_list(eval_ix) = evaluate(learning_param,env_param,theta);        
        tic;
    end
    
    %%% calculate delta
    for episode_th = 1:learning_param.train_episode_num
        % init params
        t = 0;
        score_path_sum = zeros(env_param.policy_param_num,1);
        state = reset_state(env_param);
        [a,~] = get_action_score(env_param, state, theta);  % choose initial action
        % single episode
        while(state.is_goal~=1 && t<learning_param.max_episode_length)
            state = mountain_car_step_next(state, a, env_param);   % state update
            state = judge_end(state,env_param); % judge whether finish
            [a,score] = get_action_score(env_param, state, theta);
            score_path_sum = score_path_sum + score;
            r = get_reward(state);
            delta = delta + r*score_path_sum;
            t = t+1;
        end
        total_step = total_step + t;
    end
    %%% update theta
    grad = delta;
    theta = theta - alpha*grad;
       
end



end

