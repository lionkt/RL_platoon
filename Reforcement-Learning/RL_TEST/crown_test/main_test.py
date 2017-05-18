import car_env as car_env
import numpy as np
import matplotlib.pyplot as plt

SIM_END_DISTANCE = car_env.ROAD_LENGTH - 500  # 在到达路的终点之前结束仿真
UPDATA_TIME_PER_DIDA = 0.03  # 在c++版本的仿真平台的3D工程中，取的时间步长是0.03
time_tag = 0.0


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


# 计算运动学参数
def CarList_calculate(Carlist, STARTEGEY):
    for single_car in Carlist:
        single_car.calculate(Carlist, STARTEGEY)


# 更新运动学信息
def CarList_update_info(Carlist, time_per_dida_I):
    for single_car in Carlist:
        single_car.update_car_info(time_per_dida_I)


# 根据当前的车辆数更新是否加入platoon的信息
def CarList_update_platoon_info(Carlist, des_platoon_size):
    if len(Carlist) < des_platoon_size:
        for single_car in Carlist:
            single_car.ingaged_in_platoon = False
    else:
        for single_car in Carlist:
            single_car.ingaged_in_platoon = False


if __name__ == '__main__':
    Carlist = []  # 车辆的数组
    car1 = car_env.car(
        id=0,
        role='leader',
        ingaged_in_platoon=False,
        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
        tar_speed=60.0 / 3.6,
        location=[0, 50]
    )
    car2 = car_env.car(
        id=1,
        role='follower',
        ingaged_in_platoon=False,
        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
        tar_speed=60.0 / 3.6,
        location=[0, 50]
    )
    car3 = car_env.car(
        id=2,
        role='follower',
        ingaged_in_platoon=False,
        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
        tar_speed=60.0 / 3.6,
        location=[0, 50]
    )

    while True:
        time_tag += car_env.AI_DT
        if len(Carlist) == 0:
            Carlist.append(car1)
        if time_tag >= 2 and len(Carlist) == 1:
            Carlist.append(car2)
        # if time_tag >= 6 and len(Carlist) == 2:
        #     Carlist.append(car3)
        CarList_update_platoon_info(Carlist, 2)

        # 计算运动学信息
        CarList_calculate(Carlist, 'CACC')

        # 更新运动学参数。由于c++程序的3D和CarAI的时钟不同步，需要模仿那个程序进行多轮次更新
        turns = 0
        while turns <= car_env.AI_DT:
            CarList_update_info(Carlist, UPDATA_TIME_PER_DIDA)
            turns += UPDATA_TIME_PER_DIDA

        # 终止条件判断
        car1_now_y = car1.location[1]
        print('time_tag:%.2f' % time_tag, ',now_car1_y:%.2f' % car1_now_y)
        if car1_now_y >= SIM_END_DISTANCE:
            break

    # 绘制结果
    plot_data(Carlist)
