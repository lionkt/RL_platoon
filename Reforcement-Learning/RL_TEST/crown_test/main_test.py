import car_env as car_env
import plot_funcion as my_plot
import numpy as np

SIM_END_DISTANCE = car_env.ROAD_LENGTH - 200  # 在到达路的终点之前结束仿真
UPDATA_TIME_PER_DIDA = 0.03  # 在c++版本的仿真平台的3D工程中，取的时间步长是0.03
time_tag = 0.0


# 计算运动学参数
def CarList_calculate(Carlist, STARTEGEY):
    for single_car in Carlist:
        single_car.calculate(Carlist, STARTEGEY, time_tag)


# 更新运动学信息
def CarList_update_info(Carlist, time_per_dida_I):
    for single_car in Carlist:
        single_car.update_car_info(time_per_dida_I)


# 根据build_platoon，更新是否加入platoon的信息
def CarList_update_platoon_info(Carlist, des_platoon_size, build_platoon):
    if build_platoon == False:
        for single_car in Carlist:
            single_car.ingaged_in_platoon = False
    else:
        for single_car in Carlist:
            single_car.leader = Carlist[0]
        if len(Carlist) < des_platoon_size:
            for single_car in Carlist:
                single_car.ingaged_in_platoon = False
        else:
            for single_car in Carlist:
                single_car.ingaged_in_platoon = True


if __name__ == '__main__':
    Carlist = []  # 车辆的数组
    car1 = car_env.car(
        id=0,
        role='leader',
        ingaged_in_platoon=False,
        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
        tar_speed=60.0 / 3.6,
        location=[0, 75]
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
        location=[0, 25]
    )
    car4 = car_env.car(
        id=3,
        role='follower',
        ingaged_in_platoon=False,
        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
        tar_speed=60.0 / 3.6,
        location=[0, 0]
    )
    car5 = car_env.car(
        id=4,
        role='follower',
        ingaged_in_platoon=False,
        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
        tar_speed=60.0 / 3.6,
        location=[0, -25]
    )

    if len(Carlist) == 0:
        Carlist.append(car1)
        Carlist.append(car2)
        Carlist.append(car3)
        Carlist.append(car4)
        Carlist.append(car5)


    while True:
        # 时间戳更新
        time_tag += car_env.AI_DT

        # 将新车加入车队
        # if len(Carlist) == 0:
        #     Carlist.append(car1)
        # if time_tag >= 2 and len(Carlist) == 1:
        #     Carlist.append(car2)
        # if time_tag >= 4 and len(Carlist) == 2:
        #     Carlist.append(car3)

        # 根据build_platoon，更新是否加入platoon
        CarList_update_platoon_info(Carlist, des_platoon_size=4, build_platoon=True)

        # 计算运动学信息
        CarList_calculate(Carlist, STARTEGEY='CACC')
        # CarList_calculate(Carlist, STARTEGEY='ACC')


        # 更新运动学参数。由于c++程序的3D和CarAI的时钟不同步，需要模仿那个程序进行多轮次更新
        turns = 0
        while turns <= car_env.AI_DT:
            CarList_update_info(Carlist, UPDATA_TIME_PER_DIDA)
            turns += UPDATA_TIME_PER_DIDA

        # 终止条件判断
        print('time_tag:%.2f' % time_tag, ',now_car1_y:%.2f' % car1.location[1])
        if time_tag >= 60:
            break


    # 绘制结果
    my_plot.plot_data(Carlist)
