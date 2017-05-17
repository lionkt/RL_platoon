import car_env as car_env

SIM_END_DISTANCE = car_env.ROAD_LENGTH - 100  # 在到达路的终点之前结束仿真
UPDATA_TIME_PER_DIDA = 0.03                   # 在c++版本的仿真平台的3D工程中，取的时间步长是0.03
time_tag = 0.0

if __name__ == '__main__':
    car = car_env.car(
        id=0,
        role='leader',
        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
        tar_speed=60.0 / 3.6
    )

    Carlist = []
    Carlist.append(car)

    while True:
        time_tag += car_env.AI_DT
        now_y = car.location[0, 1]
        print('time_tag:%.2f' % time_tag, ',now_y:%.2f' % now_y)
        car.calculate(Carlist)                          # 计算决策信息
        car.update_car_info(UPDATA_TIME_PER_DIDA)       # 更新运动学信息
        if now_y >= SIM_END_DISTANCE:
            break

    # 绘制结果
    car.plot_data()
