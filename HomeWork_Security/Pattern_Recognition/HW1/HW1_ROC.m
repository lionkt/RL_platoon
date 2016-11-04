clc;
lamda = 0:0.05:50;
XX = 0:0.01:1; YY = 0:0.01:1;   %用于画图
TP_in_train = zeros(length(lamda),1);
FP_in_train = zeros(length(lamda),1);
TP_in_test = zeros(length(lamda),1);
FP_in_test = zeros(length(lamda),1);

P_M = 0.5;
P_F = 0.5;
fileID = fopen('train_data.txt');
data = textscan(fileID,'%f\t%f\t%c');
fclose(fileID);
H = data{1};
W = data{2};
Gender = data{3};
X = [H W];
m = size(X,1);
Ix = find(Gender == 'm');
Male = X(Ix,:);
Female = X;
Female(Ix,:) = [];
Male = Male'; Female = Female';
X = X';
ave_M = mean(Male')';
ave_F = mean(Female')'; 
sigma_M = cov(Male')*(length(Male)-1)/length(Male);
sigma_F = cov(Female')*(length(Female)-1)/length(Female);

%% 训练数据
for turn = 1:length(lamda)
    gender_train = zeros(length(Gender),1);
    for i = 1:length(X)
        x1 = X(:,i) - ave_M;
        p1 = 1/((2*pi)*sqrt(det(sigma_M)))*exp(-1/2*x1'*inv(sigma_M)*x1);
        x2 = X(:,i) - ave_F;
        p2 = 1/((2*pi)*sqrt(det(sigma_F)))*exp(-1/2*x2'*inv(sigma_F)*x2);
        lx = p1/p2;
        gate = lamda(1,turn);
        if(lx >= gate)
            gender_train(i,1) = 1;  %判定为男性
        else
            gender_train(i,1) = 0;  %判定为女性
        end
    end
    gender_input = zeros(length(Gender),1);
    Ix = find(Gender == 'm');
    gender_input(Ix) = 1;   %定为男性

    TP_in_train(turn,1) = length(find(gender_train == 1 & gender_input == 1))/...
        length(find(gender_input == 1));
    FP_in_train(turn,1) = 1- length(find(gender_train == 0 & gender_input == 0))/...
        length(find(gender_input == 0));
    
    disp(['lamda ',num2str(gate)]);
end
figure;
plot(FP_in_train,TP_in_train,'linewidth',1.5);
hold on;
plot(XX,YY,':','linewidth',1.5);
legend('ROC','基准线');
xlabel('FP');
ylabel('TP');

%% 测试数据
fileID = fopen('dataset1.txt');
test = textscan(fileID,'%f\t%f\t%c');
fclose(fileID);   
T = [test{1} test{2}];
T = T';
gender_test = zeros(length(T),1);   
for turn = 1:length(lamda)
    for i = 1:length(T)
        x1 = T(:,i) - ave_M;
        p1 = 1/((2*pi)*sqrt(det(sigma_M)))*exp(-1/2*x1'*inv(sigma_M)*x1);
        x2 = T(:,i) - ave_F;
        p2 = 1/((2*pi)*sqrt(det(sigma_F)))*exp(-1/2*x2'*inv(sigma_F)*x2);
        lx = p1/p2;
        gate = lamda(1,turn);
        if(lx >= gate)
            gender_test(i,1) = 1;  %判定为男性
        else
            gender_test(i,1) = 0;  %判定为女性
        end
    end
    gender_input = zeros(length(test{3}),1);
    Ix = find(test{3} == 'M');
    gender_input(Ix) = 1;   %定为男性

    TP_in_test(turn,1) = length(find(gender_test == 1 & gender_input == 1))/...
        length(find(gender_input == 1));
    FP_in_test(turn,1) = 1- length(find(gender_test == 0 & gender_input == 0))/...
        length(find(gender_input == 0));
    
    disp(['lamda ',num2str(gate)]);
end
figure;
plot(FP_in_test,TP_in_test,'linewidth',1.5);
hold on;
plot(XX,YY,':','linewidth',1.5);
legend('ROC','基准线');
title('test data');
xlabel('FP');
ylabel('TP');
