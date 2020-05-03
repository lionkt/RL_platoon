import numpy as np
import pandas as pd
import os


def calculate_indicator(path):
    speed_df = pd.read_table(os.path.join(path, 'speed_data.txt'), sep=' ')
    acc_df = pd.read_table(os.path.join(path, 'acc_data.txt'), sep=' ')
    jerk_df = pd.read_table(os.path.join(path, 'jerk_data.txt'), sep=' ')
    location_df = pd.read_table(os.path.join(path, 'location_data.txt'), sep=' ')
    inter_dist_df = pd.read_table(os.path.join(path, 'inter-distance_data.txt'), sep=' ')
    col_name = ['leader'] + ['F' + str(x + 1) for x in range(speed_df.shape[1] - 1)]
    speed_df.columns = col_name
    acc_df.columns = col_name
    jerk_df.columns = col_name
    location_df.columns = col_name
    inter_dist_df.columns = ['d' + str(x) for x in range(inter_dist_df.shape[1])]
    # 定义基本参数
    Time_Step = 0.2
    Car_Length = 5
    Desired_inter_distance = 5
    MAX_V = 60 / 3.6
    TIME_TAG_UP_BOUND = 120
    ROAD_LENGTH = MAX_V * TIME_TAG_UP_BOUND
    START_LEADER_TEST_DISTANCE = ROAD_LENGTH / 1.4

    # 确定有效数据
    ix = (location_df['leader'] > START_LEADER_TEST_DISTANCE)

    # 计算rmse
    def calc_rmse(df, ix):
        # 第一列是leader，计算其他follower和leader之间的rmse
        for i in range(df.shape[1] - 1):
            tmp_df = (df['F' + str(i + 1)][ix] - df['leader'][ix]) ** 2
            sum_df = (sum_df + tmp_df) if i > 0 else tmp_df
        rmse_val = np.sqrt(sum_df.mean() / (df.shape[1] - 1))
        return rmse_val

    def calc_inter_dist_rmse(df, ix):
        for i in range(df.shape[1]):
            tmp_df = (df['d' + str(i)][ix] - Desired_inter_distance) ** 2
            sum_df = (sum_df + tmp_df) if i > 0 else tmp_df
        rmse_val = np.sqrt(sum_df.mean() / (df.shape[1]))
        return rmse_val

    def calc_error_bound_time(df):
        percent = 5e-2
        time_mean = 0.0
        for i in range(df.shape[1]):
            ix = (df['d' + str(i)]-Desired_inter_distance).abs() < Desired_inter_distance*percent
            first_time = (ix == True).idxmax() * Time_Step    # 找到第一个进入5%误差带的位置
            time_mean += first_time
        time_mean = time_mean / df.shape[1]
        return time_mean

    # 输出计算结果
    print('=============== indicator =================')
    print('speed rmse:%.3f' % calc_rmse(speed_df, ix), ',acc rmse:%.2f' % calc_rmse(acc_df, ix),
          ',inter spacing rmse:%.3f' % calc_inter_dist_rmse(inter_dist_df, ix),
          ',1st time in 5%% bound:%.3f' % calc_error_bound_time(inter_dist_df))


if __name__ == '__main__':
    calculate_indicator('./OutputImg')
