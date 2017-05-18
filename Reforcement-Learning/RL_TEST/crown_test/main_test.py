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

    plt.figure(0)
    for single_car in CarList_I:
        if max_speed_length > len(single_car.speedData):
            data = list(np.zeros(max_speed_length - len(single_car.speedData))) + single_car.speedData
        else:
            data = single_car.speedData
        plt.plot(np.arange(max_speed_length), data)
    plt.ylabel('speed')
    plt.xlabel('time_steps')
    # plt.show()

    plt.figure(1)
    for single_car in CarList_I:
        if max_acc_length > len(single_car.accData):
            data = list(np.zeros(max_acc_length - len(single_car.accData))) + single_car.accData
        else:
            data = single_car.accData
        plt.plot(np.arange(max_acc_length), data)
    plt.ylabel('acc')
    plt.xlabel('time_steps')
    # plt.show()

    # plt.figure(2)
    # for single_car in CarList_I:
    #     locationData_plt = np.array(single_car.locationData)
    #     if max_location_length > len(single_car.locationData):
    #         data = list(np.zeros(max_location_length - len(single_car.locationData))) + locationData_plt[:,
    #                                                                               1]  # 只有numpy才支持这样切片，list不支持
    #     else:
    #         data = locationData_plt[:, 1]
    #     plt.plot(np.arange(max_location_length), data)
    # plt.ylabel('location')
    # plt.xlabel('time_steps')
    plt.show()


if __name__ == '__main__':
    car1 = car_env.car(
        id=0,
        role='leader',
        ingaged_in_platoon=True,
        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
        tar_speed=60.0 / 3.6,
        location=[0, 20]
    )
    car2 = car_env.car(
        id=1,
        role='follower',
        ingaged_in_platoon=True,
        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
        tar_speed=60.0 / 3.6,
        location=[0, 5]
    )

    Carlist = []
    Carlist.append(car1)

    while True:
        time_tag += car_env.AI_DT
        car1_now_y = car1.location[1]
        print('time_tag:%.2f' % time_tag, ',now_car1_y:%.2f' % car1_now_y)
        if time_tag >= 5 and len(Carlist) == 1:
            Carlist.append(car2)

        car1.calculate(Carlist)  # 计算决策信息
        if len(Carlist) == 2:
            car2.calculate(Carlist)  # 计算决策信息

        car1.update_car_info(UPDATA_TIME_PER_DIDA)  # 更新运动学信息
        if len(Carlist) == 2:
            car2.update_car_info(UPDATA_TIME_PER_DIDA)  # 更新运动学信息

        if car1_now_y >= SIM_END_DISTANCE:
            break

    # 绘制结果
    plot_data(Carlist)
