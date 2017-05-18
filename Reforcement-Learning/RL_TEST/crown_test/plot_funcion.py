import matplotlib.pyplot as plt
import car_env as car_env
import numpy as np

# 画图函数
def plot_data(CarList_I):
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
        plt.plot(np.arange(max_speed_length), data)
    plt.ylabel('speed')
    plt.xlabel('time_steps')

    for single_car in CarList_I:
        if max_acc_length > len(single_car.accData):
            data = list(np.zeros(max_acc_length - len(single_car.accData))) + single_car.accData
        else:
            data = single_car.accData
        plt.subplot(212)
        plt.plot(np.arange(max_acc_length), data)
    plt.ylabel('acc')
    plt.xlabel('time_steps')

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
        plt.plot(np.arange(max_location_length), data[index])
        index += 1
    plt.ylabel('location')
    plt.xlabel('time_steps')

    index = 0
    for index in range(len(data) - 1):
        plt.subplot(212)
        plt.plot(np.arange(max_location_length), np.array(data[index]) - np.array(data[index + 1]) - car_env.CAR_LENGTH)
    plt.ylabel('inter-space')
    plt.xlabel('time_steps')

    plt.show()