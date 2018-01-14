function [ state ] = judge_end( state,env_param )
% judge whether reach goal
if (state.x(1) >= env_param.goal)
    state.is_goal = 1;
else 
    state.is_goal = 0;
end

end

