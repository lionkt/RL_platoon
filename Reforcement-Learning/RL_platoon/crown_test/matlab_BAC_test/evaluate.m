function [ mean_step ] = evaluate(learning_param,env_param,theta)
% evaluate now parameters
total_step = 0;
for th_ =1:learning_param.eval_episode_num
    t = 0;
    state = reset_state(env_param);
    [a,~] = get_action_score(env_param, state, theta);  % choose initial action
    % single episode
    while(state.is_goal~=1 && t<learning_param.max_episode_length)
        state = mountain_car_step_next(state, a, env_param);   % state update
        state = judge_end(state,env_param); % judge whether finish
        [a,~] = get_action_score(env_param, state, theta);
        t = t+1;
    end
    total_step = total_step + t;
end

mean_step = total_step/learning_param.eval_episode_num; %以平均的步长作为评估的标准


end

