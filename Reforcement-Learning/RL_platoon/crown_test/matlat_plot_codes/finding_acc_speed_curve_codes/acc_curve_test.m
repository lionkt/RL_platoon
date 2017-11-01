clc; clear all;
TURNS = 200;
AI_DT = 0.2;
MAX_ACC = 6;
MIN_ACC = -10;
MAX_V = 60/3.6;
TOLERENCE = AI_DT*MAX_ACC*0.05;

Max_V_Keep_Turns = TURNS*0.2;  %最大速度保持的轮次
turn = 0;
v_arr = zeros(TURNS,1);
acc_arr = zeros(TURNS,1);
distance_arr = zeros(TURNS,1);
cur_v = 0;
cur_acc = 0;
start_to_slowdown = 0;
turn = 0;   %d当前的轮次
for i=2:TURNS
    if(cur_v+TOLERENCE<MAX_V && start_to_slowdown==0) %加速阶段
        cur_acc = -MAX_ACC/(MAX_V*MAX_V)*cur_v*cur_v + MAX_ACC;
        cur_v = cur_v + AI_DT*cur_acc;
        acc_arr(i) = cur_acc;
        v_arr(i) = cur_v;
        distance_arr(i) = cur_v*AI_DT + distance_arr(i-1);  %分段匀速运动的思想
    elseif(abs(cur_v-MAX_V)<=TOLERENCE && turn<Max_V_Keep_Turns) %匀速阶段
        cur_acc = 0;
        cur_v = cur_v + AI_DT*cur_acc;
        acc_arr(i) = cur_acc;
        v_arr(i) = cur_v;
        distance_arr(i) = cur_v*AI_DT + distance_arr(i-1);
        turn = turn+1;
        if(turn==Max_V_Keep_Turns)
            start_to_slowdown=1;
        end
    elseif(start_to_slowdown==1)             %减速阶段
        cur_acc = MIN_ACC/(MAX_V*MAX_V)*cur_v*cur_v;
        cur_v = cur_v + AI_DT*cur_acc;
        acc_arr(i) = cur_acc;
        v_arr(i) = cur_v;
        distance_arr(i) = cur_v*AI_DT + distance_arr(i-1);  %分段匀速运动的思想
    end
end

figure;
plot(acc_arr);
title('acc');

figure;
plot(v_arr);
title('v');

figure;
plot(v_arr,acc_arr);
title('v-acc');

figure;
plot(distance_arr);
title('distance');
        
        
        
        
        