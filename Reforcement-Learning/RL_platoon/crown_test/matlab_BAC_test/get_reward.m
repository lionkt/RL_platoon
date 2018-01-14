function [ reward ] = get_reward( state )
% calculate reward according to now state
if state.is_goal==0
    reward = - 1;
elseif state.is_goal==1
    reward = 0;
end


end

