import matplotlib.pyplot as plt
import car_env as car_env
import numpy as np


# 画图函数核心函数
def plot_data_core(CarList_I):
    max_speed_length = 0
    max_acc_length = 0
    max_location_length = 0
    for single_car in CarList_I:
        max_speed_length = len(single_car.speedData) if max_speed_length < len(
            single_car.speedData) else max_speed_length
        max_acc_length = len(single_car.speedData) if max_acc_length < len(single_car.speedData) else max_acc_length
        max_location_length = len(single_car.locationData) if max_location_length < len(
            single_car.locationData) else max_location_length

    # plot dynamics
    plt.figure('dynamics')
    for single_car in CarList_I:
        if max_speed_length > len(single_car.speedData):
            data = list(np.zeros(max_speed_length - len(single_car.speedData))) + single_car.speedData
        else:
            data = single_car.speedData
        plt.subplot(211)
        label = 'leader' if single_car.role == 'leader' else 'car' + str(single_car.id + 1)
        if single_car.role == 'leader':
            plt.plot(np.arange(max_speed_length), data, color='red', label=label, linewidth=2)
        else:
            plt.plot(np.arange(max_speed_length), data, label=label, linewidth=1.4)
    plt.title('速度')
    plt.legend(loc=1)
    plt.ylabel('m/s')
    plt.grid(True)
    # plt.xlabel('time_steps')

    for single_car in CarList_I:
        if max_acc_length > len(single_car.accData):
            data = list(np.zeros(max_acc_length - len(single_car.accData))) + single_car.accData
        else:
            data = single_car.accData
        plt.subplot(212)
        label = 'leader' if single_car.role == 'leader' else 'car' + str(single_car.id + 1)
        if single_car.role == 'leader':
            plt.plot(np.arange(max_acc_length), data, color='red', label=label, linewidth=2)
        else:
            plt.plot(np.arange(max_acc_length), data, label=label, linewidth=1.4)
    plt.title('加速度')
    plt.legend(loc=1)
    plt.ylabel('m/s^2')
    plt.xlabel('time_steps')
    plt.grid(True)

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
        plt.subplot(211)
        label = 'leader' if single_car.role == 'leader' else 'car' + str(single_car.id + 1)
        if single_car.role == 'leader':
            plt.plot(np.arange(max_location_length), data[index], color='red', label=label, linewidth=2)
        else:
            plt.plot(np.arange(max_location_length), data[index], label=label, linewidth=1.5)
        index += 1

    plt.title('location')
    plt.legend(loc=4)
    plt.grid(True)
    plt.ylabel('m')
    # plt.xlabel('time_steps')

    index = 0
    for index in range(len(data) - 1):
        plt.subplot(212)
        label = 'car' + str(index + 1) + '-car' + str(index + 2)
        plt.plot(np.arange(max_location_length), np.array(data[index]) - np.array(data[index + 1]) - car_env.CAR_LENGTH,
                 label=label, linewidth=1.5)
    plt.title('inter-space')
    plt.legend(loc=1)
    plt.grid(True)
    plt.ylabel('m')
    plt.xlabel('time_steps')

    # plt.show()


# 画图的函数
def plot_data(CarList_I):
    plot_data_core(CarList_I)
    plt.show()

#存储画出的图像
