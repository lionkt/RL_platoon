import numpy as np
import scipy as sp

# define boundary
MAX_CAR_NUMBER = 5  # 最大的车辆数目
MIN_ACC = -10.0
MAX_ACC = 6.0
MAX_V = 60 / 3.6
TURN_MAX_V = 4.2
ROAD_LENGTH = MAX_V * 60
CAR_LENGTH = 5
LANE_WIDTH = 3.5
AI_DT = 0.2  # 信息决策的步长
UPDATA_TIME_PER_DIDA = 0.03  # 在c++版本的仿真平台的3D工程中，取的时间步长是0.03

START_LEADER_TEST_DISTANCE = ROAD_LENGTH / 2
EQUAL_TO_ZERO_SPEEED = 0.2

DES_PLATOON_INTER_DISTANCE = 5  # 车队的理想间距
ROLE_SPACE = ['leader', 'follower']
FOLLOW_STRATEGY = ['ACC', 'CACC', 'RL']

# DQN强化学习相关的变量
# action_single_step = 1  # 动作空间的步长
# ACTION_SPACE = list(np.arange(MIN_ACC, MAX_ACC, action_single_step))
TIME_TAG_UP_BOUND = 60.0

# DDPG强化学习相关的变量
STATE_DIM = 2
ACTION_DIM = 1
ACTION_BOUND = [MIN_ACC, MAX_ACC]


# define car
class car(object):
    def __init__(
            self,
            id,
            role,
            tar_interDis,
            tar_speed,
            location=None,
            ingaged_in_platoon=None,
            leader=None,
            previousCar=None,
            car_length=None
    ):
        self.id = id
        self.role = role
        self.speed = 0.0
        self.acc = 0.0
        if not location:
            self.location = np.zeros((1, 2))
        else:
            self.location = location[:]

        self.target_interDis = tar_interDis
        self.target_speed = tar_speed
        self.ingaged_in_platoon = ingaged_in_platoon if ingaged_in_platoon else False  # 默认不参加
        self.leader = leader
        self.previousCar = previousCar
        self.length = CAR_LENGTH if not car_length else car_length

        # 构建测试环境
        self.start_test = False
        self.sin_wave_clock = 0.0

        # 暂时用来存储，方便画图
        self.accData = []
        self.speedData = []
        self.locationData = []

    # 用acc-speed curve做限幅
    def __engine_speed_up_acc_curve(self, speed, p):
        acc_max = MAX_ACC
        v_max = MAX_V
        m = (v_max * p - v_max + v_max * sp.sqrt(1 - p)) / p
        k = (1 - p) * acc_max / m / m
        calcValue = -k * (speed - m) * (speed - m) + acc_max
        return calcValue

    # 用acc-speed curve做限幅
    def __engine_slow_down_acc_curve(self, speed, p):
        acc_max = MAX_ACC
        v_max = MAX_V
        m = v_max / (sp.sqrt(p) + 1)
        k = -MIN_ACC / m / m
        calcValue = k * (speed - m) * (speed - m) + MIN_ACC
        return calcValue

    # 启动运行的函数
    def __excute_foward(self):
        if self.speed >= MAX_V:
            self.acc = 0
        else:
            temp_a = car.__engine_speed_up_acc_curve(self, self.speed, p=0.3)
            self.acc = temp_a

    # 单纯计算前车和自己的距离，不含车长
    def __calc_pure_interDistance(self, previous):
        if (not previous):
            return ROAD_LENGTH - self.location[1]
        # assert  previous.__class__
        return previous.location[1] - self.location[1] - self.length / 2 - previous.length / 2

    # ACC的跟驰方法
    def __follow_car_ACC(self, pure_interval, previous):
        assert previous, 'ACC跟驰前车为空'  # 如果previous为空则报警
        v1 = self.speed  # 自己的速度
        v2 = previous.speed + AI_DT * previous.acc  # 前车的速度
        lam_para = 0.1
        epsilon = v1 - v2
        T = DES_PLATOON_INTER_DISTANCE / (1.0 * MAX_V)

        # 固定车头时距的跟驰方式
        sigma = -pure_interval + T * v1
        tem_a = -(epsilon + lam_para * sigma) / T
        # 限幅
        if (tem_a > MAX_ACC):
            self.acc = MAX_ACC
        elif (tem_a < MIN_ACC):
            self.acc = MIN_ACC
        else:
            self.acc = tem_a

    # CACC的跟驰方法
    def __follow_car_CACC(self, pure_interval, previous):
        assert previous, 'CACC跟驰前车为空'  # 如果previous为空则报警
        assert self.leader, 'CACC不存在leader'
        gap = DES_PLATOON_INTER_DISTANCE
        C_1 = 0.5
        w_n = 0.2
        xi = 1
        # 系数
        alpha_1 = 1 - C_1
        alpha_2 = C_1
        alpha_3 = -(2 * xi - C_1 * (xi + sp.sqrt(xi * xi - 1))) * w_n
        alpha_4 = - C_1 * (xi + sp.sqrt(xi * xi - 1)) * w_n
        alpha_5 = -w_n * w_n
        pre_acc = previous.acc
        leader_acc = self.leader.acc
        epsilon_i = -pure_interval + gap
        d_epsilon_i = self.speed - previous.speed
        # 核心公式
        tem_a = alpha_1 * pre_acc + alpha_2 * leader_acc + alpha_3 * d_epsilon_i + alpha_4 * (
            self.speed - previous.speed) + alpha_5 * epsilon_i
        # 限幅
        if (tem_a > MAX_ACC):
            self.acc = MAX_ACC
        elif (tem_a < MIN_ACC):
            self.acc = MIN_ACC
        else:
            self.acc = tem_a

    def __follow_car_for_platoon(self, STRATEGY, previous):
        temp_a = MAX_ACC
        if (not previous):
            # 如果前车为空，说明自己是leader
            if (self.speed <= TURN_MAX_V):
                temp_a = car.__engine_speed_up_acc_curve(self, self.speed, p=0.3)
            elif (self.speed > MAX_V):
                delta_v = np.abs(self.speed - MAX_V)
                temp_a = -car.__engine_speed_up_acc_curve(self, self.speed - delta_v, p=0.3) * 0.5
        else:
            v1 = self.speed  # 自己的速度
            v2 = previous.speed  # 前车的速度
            if (previous.acc < 0.0):
                v2 += AI_DT * previous.acc
            v1 = v1 if v1 > 0 else 0.0
            v2 = v2 if v2 > 0 else 0.0
            s = car.__calc_pure_interDistance(self, previous)

            # 根据策略选择跟驰的方式
            assert self.ingaged_in_platoon, '在follow_car_for_platoon中，ingaged_in_platoon出现了错误'
            # 如果参加了车队
            if STRATEGY == 'ACC':
                car.__follow_car_ACC(self, s, previous)
            elif STRATEGY == 'CACC':
                if (not self.leader) or (self.id == self.leader.id):
                    car.__follow_car_ACC(self, s, previous)  # 调用ACC来补救
                else:
                    car.__follow_car_CACC(self, s, previous)  # 调用正宗的CACC

        # 限幅
        if temp_a > MAX_ACC:
            self.acc = MAX_ACC
        elif temp_a < MIN_ACC:
            self.acc = MIN_ACC
        if temp_a < self.acc:
            self.acc = temp_a

    # 不参与车队的跟驰
    def __follow_car(self, previous):
        temp_a = MAX_ACC
        if not previous:
            # 如果前车为空，说明自己是leader
            if self.speed <= TURN_MAX_V:
                temp_a = car.__engine_speed_up_acc_curve(self, self.speed, p=0.3)
            elif self.speed > MAX_V:
                delta_v = sp.abs(self.speed - MAX_V)
                temp_a = -car.__engine_speed_up_acc_curve(self, self.speed - delta_v, p=0.3) * 0.5
        else:
            v1 = self.speed  # 自己的速度
            v2 = previous.speed  # 前车的速度
            if previous.acc < 0.0:
                v2 += AI_DT * previous.acc
            v1 = v1 if v1 > 0 else 0.0
            v2 = v2 if v2 > 0 else 0.0
            s = car.__calc_pure_interDistance(self, previous)
            safer_distance = DES_PLATOON_INTER_DISTANCE
            follow_dis = self.length / 4.47 * v1 + safer_distance
            s -= follow_dis
            if s <= 0.0:
                temp_a = MIN_ACC
            else:
                temp_a = 2.0 * (s / 2.0 - v1 * AI_DT) / (AI_DT * AI_DT)

            if s <= follow_dis:
                if temp_a > 0.0:
                    temp_a /= 2.0

        # 限幅
        if temp_a > MAX_ACC:
            self.acc = MAX_ACC
        elif temp_a < MIN_ACC:
            self.acc = MIN_ACC
        if temp_a < self.acc:
            self.acc = temp_a

    # 获取前车--为了简化起见，直接判断ID，目前假定车辆的是头车ID=0，然后后面的车依次递增
    def __get_previous_car(self, CarList):
        ''
        if self.id == 0:
            return None
        else:
            index = self.id - 1
            return CarList[index]

    # 构建测试场景
    def __test_scenario(self, TEST_SCENARIO, time_tag):
        if TEST_SCENARIO == 'leader_sin_wave':
            SIN_WAVE_A = 2.0
            SIN_WAVE_T = 8.0
            if self.role == 'leader':
                if self.location[1] >= START_LEADER_TEST_DISTANCE:
                    if self.start_test == False:
                        self.sin_wave_clock = time_tag
                        self.start_test = True
                    self.acc = -SIN_WAVE_A * np.sin((time_tag - self.sin_wave_clock) / SIN_WAVE_T * 2 * np.pi)
        elif TEST_SCENARIO == 'leader_stop':
            if self.role == 'leader':
                if self.location[1] >= START_LEADER_TEST_DISTANCE:
                    if self.start_test == False:
                        self.start_test = True
                    if self.speed > EQUAL_TO_ZERO_SPEEED:
                        self.acc = car.__engine_slow_down_acc_curve(self, self.speed, p=0.8)
                    else:
                        self.acc = car.__engine_slow_down_acc_curve(self, self.speed, p=0.9)

    # 车辆运动学的主函数
    def calculate(self, CarList, STARTEGEY, time_tag, action=None):
        # 存储上次的数据
        self.accData.append(self.acc)
        self.speedData.append(self.speed)
        loc = self.location[:]
        self.locationData.append(loc)

        old_acc = self.acc
        alpha = 0.6  # 动窗口的系数

        # 如果运行reinforcement-learning
        if STARTEGEY == 'RL' and self.role == 'follower':
            # 如果运行reinforcement-learning
            assert action, '在RL中输入的action为空'
            self.acc = action  # 把输入的action当作下标，从动作空间中取值
            ''
        else:
            # 非RL，或者RL下的leader
            # 启动车辆
            car.__excute_foward(self)
            # 跟驰，或者启动测试
            # test_method = 'leader_sin_wave'
            test_method = 'leader_stop'
            if self.start_test == True and self.role == 'leader':
                car.__test_scenario(self, test_method, time_tag)
            else:
                # 车辆跟驰
                precar = car.__get_previous_car(self, CarList)
                if self.ingaged_in_platoon:
                    car.__follow_car_for_platoon(self, STARTEGEY, precar)  # 先默认车队的跟驰成员采用ACC方法
                else:
                    car.__follow_car(self, precar)
                # 还是要执行一次测试，然后才能跳过follow
                car.__test_scenario(self, test_method, time_tag)

        # 减速限制函数，控制在包络线的范围内
        if self.acc < 0.0:
            low_ = car.__engine_slow_down_acc_curve(self, self.speed, p=0.6)
            if self.acc < low_ and low_ <= 0.0:
                self.acc = low_
            if self.acc < MIN_ACC:
                self.acc = MIN_ACC
            if np.abs(self.acc - MIN_ACC) <= 0.0:
                self.acc = old_acc * alpha + (1 - alpha) * self.acc  # 窗口平滑处理

        # 添加jerk限制函数
        beta = 0.7
        jerk_cur = (self.acc - old_acc) / AI_DT
        MAX_jerk = beta * MAX_ACC / AI_DT
        if np.abs(jerk_cur) > MAX_jerk:
            if self.acc <= 0.0:
                self.acc = -MAX_jerk * AI_DT + old_acc
            else:
                self.acc = MAX_jerk * AI_DT + old_acc
        # speed为零时acc不可能小于零
        if self.speed == 0:
            if self.acc < 0:
                self.acc = 0

        if self.acc < MIN_ACC:
            self.acc = MIN_ACC
        if self.acc > MAX_ACC:
            self.acc = MAX_ACC

    # 更新车辆的运动学信息
    def update_car_info(self, time_per_dida_I, action=None):
        last_acc = self.accData[-1]
        last_speed = self.speedData[-1]
        self.speed = self.speed + time_per_dida_I * self.acc
        if self.speed <= 0:
            self.speed = 0
        self.location[1] = self.location[1] + self.speed * time_per_dida_I


## car类相关的外部函数 ##
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


# 计算运动学参数
def CarList_calculate(Carlist, STARTEGEY, time_tag, action):
    for single_car in Carlist:
        single_car.calculate(Carlist, STARTEGEY, time_tag, action)


# 更新运动学信息的核心函数
def CarList_update_info_core(Carlist, time_per_dida_I):
    for single_car in Carlist:
        single_car.update_car_info(time_per_dida_I)


# 根据动作值步进更新，仅用于RL
def step_next(Carlist, time_tag, action):
    CarList_calculate(Carlist, STARTEGEY='RL', time_tag=time_tag, action=action)  # 将输入的动作用于运算
    turns = 0
    done = False
    while turns <= AI_DT:
        CarList_update_info_core(Carlist, UPDATA_TIME_PER_DIDA)
        turns += UPDATA_TIME_PER_DIDA

    # 设计终止条件
    # 1.时间到头了
    info = ''
    if time_tag >= TIME_TAG_UP_BOUND:
        info = 'time_end'
        done = True
    else:
        # 2.两个车基本上撞在一起了
        for single_car in Carlist:
            if single_car.id == 0 or single_car.role == 'leader':
                continue
            else:
                if Carlist[0].location[1] - single_car.location[1] - Carlist[
                    0].length / 2 - single_car.length / 2 <= 0.05:
                    info = 'crash'
                    done = True
                    break

    # 设计状态值
    data_list = []
    observation = []
    for single_car in Carlist:
        data_list.append(float(single_car.speed))
        data_list.append(float(single_car.location[1]))
    # observation = np.array(data_list)
    leader_v = data_list[0]
    leader_y = data_list[1]
    follower_v = data_list[2]
    follower_y = data_list[3]
    pure_interDistance = leader_y - follower_y - CAR_LENGTH / 2 - CAR_LENGTH / 2
    delta_v = leader_v - follower_v
    observation.append(delta_v)
    observation.append(pure_interDistance)
    observation = np.array(observation)
    return observation, done, info


# reward计算方法1：根据和leader的距离计算奖励
# 仿照Cooperative Adaptive Cruise Control: A Reinforcement Learning Approach的设置方法
def get_reward_table(observation):
    # 暂时只考虑1个leader+1个follower
    # leader_v = observation[0]
    # leader_y = observation[1]
    # follower_v = observation[2]
    # follower_y = observation[3]
    # pure_interDistance = leader_y - follower_y - CAR_LENGTH / 2 - CAR_LENGTH / 2
    # delta_v = leader_v - follower_v
    delta_v = observation[0]
    pure_interDistance = observation[1]
    if np.abs(pure_interDistance - DES_PLATOON_INTER_DISTANCE) <= 0.1 * DES_PLATOON_INTER_DISTANCE:
        return 50
    elif np.abs(pure_interDistance - DES_PLATOON_INTER_DISTANCE) <= 0.2 * DES_PLATOON_INTER_DISTANCE:
        return 20
    elif pure_interDistance <= DES_PLATOON_INTER_DISTANCE and pure_interDistance > DES_PLATOON_INTER_DISTANCE / 7:
        if delta_v <= 0:
            return -35
        else:
            return 0.5
    elif pure_interDistance <= DES_PLATOON_INTER_DISTANCE / 6:
        return -40
    elif pure_interDistance > 1.2 * DES_PLATOON_INTER_DISTANCE and pure_interDistance < 6 * DES_PLATOON_INTER_DISTANCE:
        if delta_v <= 0:
            return 0.5
        else:
            return -15
    else:
        if delta_v <= 0:
            return 0.5
        else:
            return -20


# reward计算方法2：计算单步的奖励
def get_reward_function(observation):
    # 暂时只考虑1个leader+1个follower
    # leader_v = observation[0]
    # leader_y = observation[1]
    # follower_v = observation[2]
    # follower_y = observation[3]
    # pure_interDistance = leader_y - follower_y - CAR_LENGTH / 2 - CAR_LENGTH / 2
    # delta_v = leader_v - follower_v
    delta_v = observation[0]
    pure_interDistance = observation[1]
    # r1 = (pure_interDistance - DES_PLATOON_INTER_DISTANCE) / DES_PLATOON_INTER_DISTANCE  # 由距离产生
    # r2 = -(leader_v - follower_v) / (leader_v + 0.1)  # 由速度产生
    # return r1 * 0.01 + r2 * 0.2

    # r1 = (45 - abs(pure_interDistance)) / 45 - DES_PLATOON_INTER_DISTANCE  # 由距离产生
    # r2 = (leader_v - abs(pure_interDistance)) / (leader_v + 0.1)  # 由速度产生
    # return r1 * 0.005 + r2 * 0.01

    # 双曲线函数的组合
    r1 = 0.0
    r2 = 0.0
    MAX_pure_distance = 10
    MAX_pure_v = 3.5
    if pure_interDistance <= DES_PLATOON_INTER_DISTANCE:
        r1 = 1 / (np.abs(pure_interDistance - DES_PLATOON_INTER_DISTANCE) + 0.05) - 1 / (pure_interDistance + 0.04)
    elif pure_interDistance <= MAX_pure_v:
        r1 = 1 / (np.abs(pure_interDistance - DES_PLATOON_INTER_DISTANCE) + 0.05) - 1 / (
            np.abs(pure_interDistance - MAX_pure_distance) + 0.04)
    else:
        r1 = 1 / (np.abs(MAX_pure_distance - DES_PLATOON_INTER_DISTANCE) + 0.05) - 1 / (
            np.abs(MAX_pure_distance - MAX_pure_distance) + 0.04)

    if delta_v <= -MAX_pure_v:
        r2 = 1 / (np.abs(-MAX_pure_v - 0.0) + 0.05) - 1 / (np.abs(-MAX_pure_v + MAX_pure_v) + 0.04)
    elif delta_v <= 0.0:
        r2 = 1 / (np.abs(delta_v - 0.0) + 0.05) - 1 / (np.abs(delta_v + MAX_pure_v) + 0.04)
    elif delta_v <= MAX_pure_v:
        r2 = 1 / (np.abs(delta_v - 0.0) + 0.05) - 1 / (np.abs(delta_v - MAX_pure_v) + 0.04)
    else:
        r2 = 1 / (np.abs(MAX_pure_v - 0.0) + 0.05) - 1 / (np.abs(MAX_pure_v - MAX_pure_v) + 0.04)
    return r1 * 0.055 + r2 * 0.045

    # 分段线性函数的组合
    # r1 = 0.0
    # r2 = 0.0
    # MAX_pure_distance = 40
    # MAX_pure_v = 5


# 初始化状态值
def reset(CarList):
    obs_list = []
    obs = []
    for single_car in CarList:
        if single_car.id == 0 or single_car.role == 'leader':
            obs_list.append(0)
            obs_list.append(50)
        else:
            obs_list.append(0)
            obs_list.append(25)
    leader_v = obs_list[0]
    leader_y = obs_list[1]
    follower_v = obs_list[2]
    follower_y = obs_list[3]
    pure_interDistance = leader_y - follower_y - CAR_LENGTH / 2 - CAR_LENGTH / 2
    delta_v = leader_v - follower_v
    obs.append(delta_v)
    obs.append(pure_interDistance)
    return np.array(obs)
