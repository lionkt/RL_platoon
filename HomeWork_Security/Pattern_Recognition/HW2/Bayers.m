clc;
clear all;
P_Male = 0.5;   %男生先验概率
P_Female = 0.5; %女生先验概率
fileID = fopen('dataset3.txt');
data = textscan(fileID,'%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%c');
fclose(fileID);
Gender = data{end};
X = cell2mat(data(1,1:end-1));  %一行是一个数据样本点，还未转置
population = size(X,1);
totalDim = size(X,2);
% randomRank = randperm(population);
% X = X(randomRank,:);
% Gender = Gender(randomRank,:);
X_small = zeros(20,totalDim);    %构造10男+10女的训练样本
IX = find(Gender == 'M'|Gender == 'm');
for i = 1:10
    X_small(i,:) = X(IX(i,1),:);
end
IX = find(Gender == 'F'|Gender == 'f');
for i = 1:10
    X_small(i+10,:) = X(IX(i,1),:);
end

while(true)
    Mode = input('请选择训练集的样本数目：\n 1： 10男+10女    2：全部训练集    q：退出\n','s');
    if(Mode ~= '1' && Mode ~= '2')
        disp('退出');
        break;
    else
        if(Mode == '1')
            Male = X_small(1:10,:);
            Femeale = X_small(11:20,:);
            Male = Male'; Female = Female'; %转置一下
        elseif(Mode == '2')
            IX = find(Gender == 'M'|Gender == 'm');
            Male = X(IX,:);
            Female = X;
            Female(IX,:) = [];
            Male = Male'; Female = Female'; %转置一下
        end
        %% 用全部特征训练
        ave_M = mean(Male')';
        ave_F = mean(Female')';
        sigma_M = cov(Male')*(length(Male)-1)/length(Male);
        sigma_F = cov(Female')*(length(Female)-1)/length(Female);
        W_M = -inv(sigma_M)/2;
        W_F = -inv(sigma_F)/2;
        w_M = inv(sigma_M)*ave_M;
        w_F = inv(sigma_F)*ave_F;
        w0_M = -1/2*ave_M'*inv(sigma_M)*ave_M - 1/2*log(det(sigma_M)) + log(P_Male);
        w0_F = -1/2*ave_F'*inv(sigma_F)*ave_F - 1/2*log(det(sigma_F)) + log(P_Female);

        %% 用第3列和第5列特征训练
        Male_spe = [Male(3,:);Male(5,:)];   
        Female_spe = [Female(3,:);Female(5,:)];
        ave_M_spe = mean(Male_spe')';
        ave_F_spe = mean(Female_spe')';
        sigma_M_spe = cov(Male_spe')*(length(Male_spe)-1)/length(Male_spe);
        sigma_F_spe = cov(Female_spe')*(length(Female_spe)-1)/length(Female_spe);
        W_M_spe = -inv(sigma_M_spe)/2;
        W_F_spe = -inv(sigma_F_spe)/2;
        w_M_spe = inv(sigma_M_spe)*ave_M_spe;
        w_F_spe = inv(sigma_F_spe)*ave_F_spe;
        w0_M_spe = -1/2*ave_M_spe'*inv(sigma_M_spe)*ave_M_spe...
            - 1/2*log(det(sigma_M_spe)) + log(P_Male);
        w0_F_spe = -1/2*ave_F_spe'*inv(sigma_F_spe)*ave_F_spe...
            - 1/2*log(det(sigma_F_spe)) + log(P_Female);

        %% 读取测试集数据
        fileID = fopen('dataset4.txt');
        test = textscan(fileID,'%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%c');
        fclose(fileID);
        T = cell2mat(test(1,1:end-1));
        T = T';

        %% 测试集_全部特征训练的结果
        gender_test = zeros(length(T),1);   %存储分类的结果
        for i = 1:length(T)
            g1 = T(:,i)'*W_M*T(:,i) + w_M'*T(:,i) + w0_M;
            g2 = T(:,i)'*W_F*T(:,i) + w_F'*T(:,i) + w0_F;
            if(g1 >= g2)
                gender_test(i,1) = 1;   %定为男性
            else
                gender_test(i,1) = 0;   %定为女性
            end
        end
        gender_input = zeros(length(test{end}),1);
        IX = find(test{end} == 'M'|test{end} == 'm');
        gender_input(IX) = 1;
        Test_Err = gender_input - gender_test;
        test_err = length(find(Test_Err ~= 0))/length(Test_Err)*100;
        disp(['测试集_全部特征训练的结果的误差为',num2str(test_err),' %']);

        %% 测试集_第3列和第5列特征训练的结果
        T_spe = [T(3,:); T(5,:)];
        gender_test = zeros(length(T_spe),1);   %存储分类的结果
        for i = 1:length(T)
            g1 = T_spe(:,i)'*W_M_spe*T_spe(:,i) + w_M_spe'*T_spe(:,i) + w0_M_spe;
            g2 = T_spe(:,i)'*W_F_spe*T_spe(:,i) + w_F_spe'*T_spe(:,i) + w0_F_spe;
            if(g1 >= g2)
                gender_test(i,1) = 1;   %定为男性
            else
                gender_test(i,1) = 0;   %定为女性
            end
        end
        gender_input = zeros(length(test{end}),1);
        IX = find(test{end} == 'M'|test{end} == 'm');
        gender_input(IX) = 1;
        Test_Err = gender_input - gender_test;
        test_err = length(find(Test_Err ~= 0))/length(Test_Err)*100;
        disp(['测试集_第3列和第5列特征训练的结果的误差为',num2str(test_err),' %']);
        
    end
end













