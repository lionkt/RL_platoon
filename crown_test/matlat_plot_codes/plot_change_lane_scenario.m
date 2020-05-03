clc;
clear;
% Output Format: time_tag,ID,x,y,v_x,v_y,max_v,acc_x,acc_y,cur_lane,from_lane,to_lane
% Car lenght: 5m, lane width: 3.5m
%%%% rl
improve_root_path = '../Data_thesis/lane_change/trafficFlow_CACC/';
improve_file_name = [improve_root_path,'change_lane_scenario_data.txt'];
improve_fileID = fopen(improve_file_name);
improve_data = textscan(improve_fileID,'%f %d %f %f %f %f %f %f %f %d %d %d','Delimiter',',','HeaderLines',7);
fclose(improve_fileID);
improve_time_tag_list = improve_data{1}; improve_time_tag_unique = unique(improve_time_tag_list);
improve_ID_list = improve_data{2}; improve_ID_unqiue = unique(improve_ID_list);
improve_loc_x_list = improve_data{3}; improve_loc_y_list = improve_data{4}; 
improve_v_x_list = improve_data{5}; improve_v_y_list = improve_data{6};
improve_max_v_lsit = improve_data{7};
improve_acc_x_list = improve_data{8};improve_acc_y_list = improve_data{9};
improve_cur_lane_list = improve_data{10}; improve_from_lane_list = improve_data{11}; improve_to_lane_list = improve_data{12};

improve_file_name = [improve_root_path,'platoon_spacing.txt'];
improve_fileID = fopen(improve_file_name);
improve_platoon_data = textscan(improve_fileID,'%f %d %f','Delimiter',',','HeaderLines',6);
fclose(improve_fileID);
improve_platoon_time_tag_list = improve_platoon_data{1};
improve_platoon_time_tag_unique = unique(improve_platoon_time_tag_list);
improve_platoon_ID_list = improve_platoon_data{2};
improve_platoon_mean_spacing = improve_platoon_data{3};


%%%% acc
basic_root_path = '../Data_thesis/lane_change/trafficFlow_ACC/';
basic_file_name = [basic_root_path,'change_lane_scenario_data.txt'];
basic_fileID = fopen(basic_file_name);
basic_data = textscan(basic_fileID,'%f %d %f %f %f %f %f %f %f %d %d %d','Delimiter',',','HeaderLines',7);
fclose(basic_fileID);
basic_time_tag_list = basic_data{1}; basic_time_tag_unique = unique(basic_time_tag_list);
basic_ID_list = basic_data{2}; basic_ID_unqiue = unique(basic_ID_list);
basic_loc_x_list = basic_data{3}; basic_loc_y_list = basic_data{4}; 
basic_v_x_list = basic_data{5}; basic_v_y_list = basic_data{6};
basic_max_v_lsit = basic_data{7};
basic_acc_x_list = basic_data{8};basic_acc_y_list = basic_data{9};
basic_cur_lane_list = basic_data{10}; basic_from_lane_list = basic_data{11}; basic_to_lane_list = basic_data{12};

basic_file_name = [basic_root_path,'platoon_spacing.txt'];
basic_fileID = fopen(basic_file_name);
basic_platoon_data = textscan(basic_fileID,'%f %d %f','Delimiter',',','HeaderLines',6);
fclose(basic_fileID);
basic_platoon_time_tag_list = basic_platoon_data{1};
basic_platoon_time_tag_unique = unique(basic_platoon_time_tag_list);
basic_platoon_ID_list = basic_platoon_data{2};
basic_platoon_mean_spacing = basic_platoon_data{3};


%% process
% select_id = 10;
% ix1 = (improve_ID_list==select_id);
% figure;
% subplot(231);
% plot(improve_loc_x_list(ix1), improve_loc_y_list(ix1));
% subplot(232);
% plot(time_tag_list(ix),v_x_list(ix));
% subplot(233);
% plot(time_tag_list(ix),acc_x_list(ix));
% subplot(234);
% plot(time_tag_list(ix),v_y_list(ix));
% subplot(235);
% plot(time_tag_list(ix),acc_y_list(ix));

improve_mean_v = zeros(size(improve_time_tag_unique));
for i=1:length(improve_time_tag_unique)
   temp_ix = (improve_time_tag_list == improve_time_tag_unique(i));
   improve_mean_v(i) = mean(improve_v_y_list(temp_ix));
end

basic_mean_v = zeros(size(basic_time_tag_unique));
for i=1:length(basic_time_tag_unique)
   temp_ix = (basic_time_tag_list == basic_time_tag_unique(i));
   basic_mean_v(i) = mean(basic_v_y_list(temp_ix));
end

figure;
plot(improve_time_tag_unique, improve_mean_v); hold on;
plot(basic_time_tag_unique, basic_mean_v);
legend('improve', 'basic');

%% 展示统计信息
% ix = (improve_platoon_ID_list==0);
% plot(improve_platoon_time_tag_list(ix),-improve_platoon_mean_spacing(ix)); hold on;
% ix = (basic_platoon_ID_list==0);
% plot(basic_platoon_time_tag_list(ix),-basic_platoon_mean_spacing(ix));

improve_mean_spacing = zeros(size(improve_platoon_time_tag_unique));
for i=1:length(improve_platoon_time_tag_unique)
   temp_ix = (improve_platoon_time_tag_list == improve_platoon_time_tag_unique(i));
   improve_mean_spacing(i) = mean(improve_platoon_mean_spacing(temp_ix));
end

basic_mean_spacing = zeros(size(basic_platoon_time_tag_unique));
for i=1:length(basic_platoon_time_tag_unique)
   temp_ix = (basic_platoon_time_tag_list == basic_platoon_time_tag_unique(i));
   basic_mean_spacing(i) = mean(basic_platoon_mean_spacing(temp_ix));
end

figure;
plot(improve_platoon_time_tag_unique, improve_mean_spacing); hold on;
plot(basic_platoon_time_tag_unique, basic_mean_spacing);

steady_time_tag = 500;
ix = (improve_time_tag_unique>steady_time_tag);
improve_v_y_mean = mean(improve_mean_v(ix));
improve_var = var(improve_mean_v(ix));
ix = (basic_time_tag_unique>steady_time_tag);
basic_v_y_mean = mean(basic_mean_v(ix));
basic_var = var(basic_mean_v(ix));

ix = (improve_platoon_time_tag_unique>steady_time_tag);
improve_platoon_spacing_mean = mean(improve_mean_spacing(ix));
improve_platoon_spacing_var = var(improve_mean_spacing(ix));
ix = (basic_platoon_time_tag_unique>steady_time_tag);
basic_platoon_spacing_mean = mean(basic_mean_spacing(ix));
basic_platoon_spacing_var = var(basic_mean_spacing(ix));

disp(['RL platoon_spacing mean:(t>',num2str(steady_time_tag),') = ',num2str(improve_platoon_spacing_mean)]);
disp(['RL platoon_spacing var:(t>',num2str(steady_time_tag),') = ',num2str(improve_platoon_spacing_var)]);
disp(['ACC platoon_spacing mean:(t>',num2str(steady_time_tag),') = ',num2str(basic_platoon_spacing_mean)]);
disp(['ACC platoon_spacing var:(t>',num2str(steady_time_tag),') = ',num2str(basic_platoon_spacing_var)]);





