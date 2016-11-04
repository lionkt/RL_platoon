clc;
P_M = 0.5;  %男生的先验概率
P_F = 0.5;  %女生的先验概率
fileID = fopen('dataset1.txt');
data = textscan(fileID,'%f\t%f\t%c');
H = data{1};
W = data{2};
Gender = data{3};
X = [H W];
m = size(X,1);
Ix = find(Gender == 'M');
Male = X(Ix,:);
Female = X;
Female(Ix,:) = [];
Male = Male'; Female = Female';
X = X';
ave_M = mean(Male')';
ave_F = mean(Female')'; 
sigma_M = cov(Male')*(length(Male)-1)/length(Male);
sigma_F = cov(Female')*(length(Female)-1)/length(Female);
fclose(fileID);

lam12_arr = 1:1:5;
lam21_arr = 6*ones(1,5) - lam12_arr;
lam11 = 0; lam22 = 0;
err_in_train = zeros(1,length(lam12_arr));
err_in_test = zeros(1,length(lam12_arr));
%% 训练集上的测试
for turn = 1:5
    lam12 = lam12_arr(1,turn);
    lam21 = lam21_arr(1,turn);
    gender_train = zeros(length(Gender),1);
    for i = 1:length(X)
        x1 = X(:,i) - ave_M;
        p1 = 1/((2*pi)*sqrt(det(sigma_M)))*exp(-1/2*x1'*inv(sigma_M)*x1);
        x2 = X(:,i) - ave_F;
        p2 = 1/((2*pi)*sqrt(det(sigma_F)))*exp(-1/2*x2'*inv(sigma_F)*x2);
        lx = p1/p2;
        gate = P_F*(lam12-lam22)/P_M/(lam21-lam11);
        if(lx >= gate)
            gender_train(i,1) = 1;  %判定为男性
        else
            gender_train(i,1) = 0;  %判定为女性
        end
    end
    gender_input = zeros(length(Gender),1);
    Ix = find(Gender == 'M');
    gender_input(Ix) = 1;
    Train_Err = gender_input - gender_train;
    train_err = length(find(Train_Err ~= 0))/length(Train_Err)*100;
%     disp(['训练集的误差为',num2str(train_err),' %']);
    err_in_train(1,turn) = train_err;
end


%% 在测试集上进行测试
fileID = fopen('train_data.txt');
test = textscan(fileID,'%f\t%f\t%c');
fclose(fileID);   
T = [test{1} test{2}];
T = T';
gender_test = zeros(length(T),1);    
for turn = 1:5
    lam12 = lam12_arr(1,turn);
    lam21 = lam21_arr(1,turn);
    for i = 1:length(T)
        x1 = T(:,i) - ave_M;
        p1 = 1/((2*pi)*sqrt(det(sigma_M)))*exp(-1/2*x1'*inv(sigma_M)*x1);
        x2 = T(:,i) - ave_F;
        p2 = 1/((2*pi)*sqrt(det(sigma_F)))*exp(-1/2*x2'*inv(sigma_F)*x2);
        lx = p1/p2;
        gate = P_F*(lam12-lam22)/P_M/(lam21-lam11);
        if(lx >= gate)
            gender_test(i,1) = 1;  %判定为男性
        else
            gender_test(i,1) = 0;  %判定为女性
        end
    end
    gender_input = zeros(length(test{3}),1);
    Ix = find(test{3} == 'm');
    gender_input(Ix) = 1;
    Test_Err = gender_input - gender_test;
    test_err = length(find(Test_Err ~= 0))/length(Test_Err)*100;
    % disp(['测试集的误差为',num2str(test_err),' %']);
    err_in_test(1,turn) = test_err;
end
    
    
    
    

