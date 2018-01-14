function [ reward ] = get_reward( state )
% calculate reward according to now state
reward = state.is_goal - 1;

end

