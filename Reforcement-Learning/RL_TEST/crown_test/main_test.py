import car_env as car_env
import numpy as np

SIM_END_DISTANCE = car_env.ROAD_LENGTH - 100  # 在到达路的终点之前结束仿真
UPDATA_TIME_PER_DIDA = 0.03  # 在c++版本的仿真平台的3D工程中，取的时间步长是0.03
time_tag = 0.0

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
        if len(Carlist)==2:
            car2.calculate(Carlist)  # 计算决策信息

        car1.update_car_info(UPDATA_TIME_PER_DIDA)  # 更新运动学信息
        if len(Carlist)==2:
            car2.update_car_info(UPDATA_TIME_PER_DIDA)  # 更新运动学信息

        if car1_now_y >= SIM_END_DISTANCE:
            break

    # 绘制结果
    car1.plot_data()
    car2.plot_data()
