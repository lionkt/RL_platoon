import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import car_env_DDPG_3cars  as car_env_3_car
import numpy as np
from scipy.fftpack import fft, ifft


# 画图函数核心函数
def plot_data_core(CarList_I, write_flag):
    max_speed_length = 0
    max_acc_length = 0
    max_location_length = 0
    for single_car in CarList_I:
        max_speed_length = len(single_car.speedData) if max_speed_length < len(
            single_car.speedData) else max_speed_length
        max_acc_length = len(single_car.speedData) if max_acc_length < len(single_car.speedData) else max_acc_length
        max_location_length = len(single_car.locationData) if max_location_length < len(
            single_car.locationData) else max_location_length

    leader_color = 'purple'

    # plot dynamics
    plt.figure('dynamics')
    write_buffer = []
    for single_car in CarList_I:
        if max_speed_length > len(single_car.speedData):
            data = list(np.zeros(max_speed_length - len(single_car.speedData))) + single_car.speedData
        else:
            data = single_car.speedData
        plt.subplot(311)
        label = 'leader' if single_car.role == 'leader' else 'car' + str(single_car.id + 1)
        if single_car.role == 'leader':
            plt.plot(np.arange(max_speed_length), data, color=leader_color, label=label, linewidth=2)
        else:
            plt.plot(np.arange(max_speed_length), data, label=label, linewidth=2)
        # 把数据写出去
        if write_flag:
            write_buffer.append(data)
    plt.title('speed')
    plt.legend(loc=4)
    plt.ylabel('m/s')
    plt.grid(True)
    if write_flag:
        write_buffer = np.array(write_buffer).transpose()
        np.savetxt('./OutputImg/speed_data.txt', write_buffer)
        print('====speed data has been written=====')
    # plt.xlabel('time_steps')

    write_buffer = []
    for single_car in CarList_I:
        if max_acc_length > len(single_car.accData):
            data = list(np.zeros(max_acc_length - len(single_car.accData))) + single_car.accData
        else:
            data = single_car.accData
        plt.subplot(312)
        label = 'leader' if single_car.role == 'leader' else 'car' + str(single_car.id + 1)
        if single_car.role == 'leader':
            plt.plot(np.arange(max_acc_length), data, color=leader_color, linewidth=2)
        else:
            plt.plot(np.arange(max_acc_length), data, linewidth=1.8)
        if write_flag:
            write_buffer.append(data)
    plt.title('acceleration')
    plt.legend(loc=1)
    plt.ylabel('m/s^2')
    # plt.xlabel('time_steps')
    plt.grid(True)
    if write_flag:
        write_buffer = np.array(write_buffer).transpose()
        np.savetxt('./OutputImg/acc_data.txt', write_buffer)
        print('====acc data has been written=====')

    write_buffer = []
    for single_car in CarList_I:
        if max_acc_length > len(single_car.accData):
            data = list(np.zeros(max_acc_length - len(single_car.accData))) + single_car.accData
        else:
            data = single_car.accData
        plt.subplot(313)
        label = 'leader' if single_car.role == 'leader' else 'car' + str(single_car.id + 1)
        data2 = np.array(data[1: ])
        data = np.array(data[ :-1])
        if single_car.role == 'leader':
            plt.plot(np.arange(max_acc_length-1), (data2-data) / car_env_3_car.AI_DT, color=leader_color, linewidth=2)
        else:
            plt.plot(np.arange(max_acc_length-1), (data2-data) / car_env_3_car.AI_DT, linewidth=1.8)
        if write_flag:
            write_buffer.append(list((data2-data) / car_env_3_car.AI_DT))
    plt.title('jerk')
    plt.legend(loc=1)
    plt.ylabel('m/s^3')
    plt.xlabel('time_steps')
    plt.grid(True)
    if write_flag:
        write_buffer = np.array(write_buffer).transpose()
        np.savetxt('./OutputImg/jerk_data.txt', write_buffer)
        print('====jerk data has been written=====')

    out_png = './OutputImg/dynamics.png'    # save file
    plt.savefig(out_png, dpi=300)


    # plot location
    plt.figure('location')
    data = []
    index = 0
    for single_car in CarList_I:
        locationData_plt = np.array(single_car.locationData)
        if max_location_length > len(single_car.locationData):
            data.append(list(np.zeros(max_location_length - len(single_car.locationData))) + list(
                locationData_plt[:, 1]))  # 只有numpy才支持这样切片，list不支持
        else:
            data.append(list(locationData_plt[:, 1]))
        # plt.subplot(211)
        # label = 'leader' if single_car.role == 'leader' else 'car' + str(single_car.id + 1)
        # if single_car.role == 'leader':
        #     plt.plot(np.arange(max_location_length), data[index], color=leader_color, label=label, linewidth=2)
        # else:
        #     plt.plot(np.arange(max_location_length), data[index], label=label, linewidth=1.5)
        index += 1

    # plt.title('location')
    # plt.legend(loc=4)
    # plt.grid(True)
    # plt.ylabel('m')
    # plt.xlabel('time_steps')
    write_buffer = []
    if write_flag:
        write_buffer = np.array(data).transpose()
        np.savetxt('./OutputImg/location_data.txt', write_buffer)
        print('====location data has been written=====')

    index = 0
    plot_desired_value_flag = True
    inter_distance_list = []
    write_buffer = []
    for index in range(len(data) - 1):
        plt.subplot(211)
        label = 'car' + str(index + 1) + '-car' + str(index + 2)
        inter_distance_list.append(np.array(data[index]) - np.array(data[index + 1]) - car_env_3_car.CAR_LENGTH)
        plt.plot(np.arange(max_location_length), inter_distance_list[index], label=label, linewidth=1.5)
        if write_flag:
            write_buffer.append(inter_distance_list[index])

    if plot_desired_value_flag:
        length_inter_space = list(np.arange(0,300,1))
        desired_inter_space = list(car_env_3_car.DES_PLATOON_INTER_DISTANCE * np.ones(len(length_inter_space)))
        plt.plot(length_inter_space, desired_inter_space,'--',linewidth=1.5,label='desired inter-space')
    plt.title('inter-space')
    plt.legend(loc=1)
    plt.grid(True)
    plt.ylabel('m')
    plt.xlabel('time_steps')
    if write_flag:
        write_buffer = np.array(write_buffer).transpose()
        np.savetxt('./OutputImg/inter-distance_data.txt', write_buffer)
        print('====inter-distance data has been written=====')


    index = 0
    # correction = car_env.DES_PLATOON_INTER_DISTANCE
    correction = 0.0 #
    for index in range(len(inter_distance_list) - 1):
        inter_distance_1 = inter_distance_list[index]
        inter_distance_2 = inter_distance_list[index + 1]
        label = 'space' + str(index + 2) + '/space' + str(index + 1)
        yf = np.abs(fft(inter_distance_2 - correction) / fft(inter_distance_1 - correction))

        # yf_bode = 20 * np.log10(yf)
        # yf_norm = yf_bode

        yf_norm = yf

        yf_norm_half = yf_norm[range(int(len(yf_norm) / 2))]
        xf = np.arange(len(yf_norm))
        xf_half = xf[range(int(len(xf) / 2))]

        plt.subplot(212)
        plt.plot(xf_half, yf_norm_half, label=label, linewidth=2)
        # plt.xscale('log')       # 以对数为轴画图

    plt.title('inter-space-error')
    plt.legend(loc=1)
    plt.grid(True)
    plt.ylabel('frequency-amplitude')
    plt.xlabel('time_steps')

    out_png = './OutputImg/location.png'  # save file
    plt.savefig(out_png, dpi=300)

    # 添加时间位移曲线
    plt.figure('location-time')
    for index in range(len(data)):
        label = 'car' + str(index + 1)
        plt.plot(np.arange(max_location_length), data[index], label=label, linewidth=1.5)
    plt.title('inter-space')
    plt.legend(loc=1)
    plt.grid(True)
    plt.ylabel('y-position')
    plt.xlabel('time_steps')
    out_png = './OutputImg/location-time.png'  # save file
    plt.savefig(out_png, dpi=300)

# 画图的函数
def plot_data(CarList_I, write_flag = None):
    plot_data_core(CarList_I, write_flag = write_flag)
    plt.show()

# 存储画出的图像
