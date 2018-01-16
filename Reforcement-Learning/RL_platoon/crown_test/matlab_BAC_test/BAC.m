function [ mean_step_list, calc_time_list, theta  ] = BAC( learning_param, env_param )
%%% Bayesian actor-critic method for RL problem
% init
alpha = learning_param.alpha_init;
theta = zeros(env_param.policy_param_num,1);
output_num = floor(learning_param.max_update_num / learning_param.eval_interval) + 1;
mean_step_list = zeros(output_num, 1);
calc_time_list = zeros(output_num, 1);

%% begin calculation
tic;
alpha_schedule = learning_param.alpha_init * ...
    (learning_param.alp_update_param./ (learning_param.alp_update_param ...
    +  (1:(learning_param.max_update_num+1) - 1)));

for update_th = 1:learning_param.max_update_num
    if mod(update_th, learning_param.eval_interval)==0
        disp(['BAC finish ',num2str(update_th/learning_param.max_update_num*100),'% of ',num2str(learning_param.max_update_num)]);
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
    
    G = sparse(env_param.policy_param_num, env_param.policy_param_num); %fisher information matrix
    episodes_store_cell = cell(learning_param.train_episode_num,3); % record list
    
    %%% calculate delta
    for episode_th = 1:learning_param.train_episode_num
        t = 0;
        episode_states = []; % BAC record
        episode_scores = []; % BAC record
        state = reset_state(env_param);
        [a,score] = get_action_score(env_param, state, theta);
        score = sparse(score);
        while (state.is_goal~=1 && t < learning_param.max_episode_length)
            state = mountain_car_step_next(state, a, env_param);   % state update
            state = judge_end(state,env_param); % judge whether finish
            G = G + (score * score');   % update G
            episode_states = [episode_states, state]; % BAC record
            episode_scores = [episode_scores, score]; % BAC record
            [a,score] = get_action_score(env_param, state, theta);
            score = sparse(score);
            t = t + 1;
        end
        %%% store data
        episodes_store_cell{episode_th,1} = episode_states;
        episodes_store_cell{episode_th,2} = episode_scores;
        episodes_store_cell{episode_th,3} = t;
    end
    %%% parameters update
    G = G + 1e-6*speye(size(G));
    grad_BAC = BAC_grad(episodes_store_cell, G,env_param,learning_param);   % calculate BAC gradient
    if learning_param.alp_schedule
        alpha = alpha_schedule(update_th);      % use changing alpha
    else
        alpha = learning_param.alpha_init;
    end
    theta = theta - (alpha * grad_BAC);
        
end




end

