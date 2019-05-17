clc;
clear all;

% syms p MAX_V MAX_ACC m
% m = solve(p*m*m +(1-p)*2*m*MAX_V-(1-p)*MAX_V*MAX_V,m);    %SPEE UP

AI_DT = 0.2;
MAX_ACC = 6;
MIN_ACC = -10;
MAX_V = 60/3.6;
TOLERENCE = AI_DT*MAX_ACC*0.05;

p = [0.5:0.1:1];
m_2 = (MAX_V*p - MAX_V + MAX_V*(1 - p).^(1/2))./p;  %正的根
% m_1 = -(3*(10000/9 - (10000*p)/9).^(1/2) - 100*p + 100)./(6*p); %speed up
% m_2 = (100*p + 3*(10000/9 - (10000*p)/9).^(1/2) - 100)./(6*p);
IX = find(m_2==0);
m_2(IX) = [];
p(IX) = [];
k = (1-p)*MAX_ACC./(m_2.*m_2);

%% 绘制acc-v curve
v = [0:0.05:MAX_V];
acc = zeros(length(m_2),length(v));
for i=1:length(m_2)
    acc(i,:) = -k(i)*(v-m_2(i)).*(v-m_2(i))+MAX_ACC;
end

figure;
for i=1:length(m_2)
    plot(v, acc(i,:),'linewidth',1.5);
    hold on;
end
xlabel('speed(m/s)'); ylabel('acc(m/s^-2)');
title('speed-acc curve');




