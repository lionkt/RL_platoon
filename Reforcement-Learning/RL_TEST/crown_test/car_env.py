import numpy as np
import scipy as sp

# define boundary
MAX_CAR_NUMBER = 5  # 最大的车辆数目
MIN_ACC = -10.0
MAX_ACC = 6.0
MAX_V = 60 / 3.6
TURN_MAX_V = 4.2
ROAD_LENGTH = MAX_V * 100
CAR_LENGTH = 5
LANE_WIDTH = 3.5
AI_DT = 0.1

DES_PLATOON_INTER_DISTANCE = 5  # 车队的理想间距
ROLE_SPACE = ['leader', 'follower']
FOLLOW_STRATEGY = ['ACC', 'CACC', 'RL']


# define car
class car(object):
    def __init__(
            self,
            id,
            role,
            tar_interDis,
            tar_speed,
            ingaged_in_platoon=None,
            leader=None,
            previousCar=None,
            car_length=None
    ):
        self.id = id
        self.role = role
        self.speed = 0.0
        self.acc = 0.0
        self.location = np.zeros((1, 2))
        self.target_interDis = tar_interDis
        self.target_speed = tar_speed
        self.ingaged_in_platoon = ingaged_in_platoon
        self.leader = leader
        self.previousCar = previousCar
        self.length = CAR_LENGTH if not car_length else car_length

    # 用acc-speed curve做限幅
    def __engine_speed_up_acc_curve(self, p):
        acc_max = MAX_ACC
        v_max = MAX_V
        m = (v_max * p - v_max + v_max * sp.sqrt(1 - p)) / p
        k = (1 - p) * acc_max / m / m
        return -k * (self.speed - m) * (self.speed - m) + acc_max

    # 用acc-speed curve做限幅
    def __engine_slow_down_acc_curve(self, p):
        acc_max = MAX_ACC
        v_max = MAX_V
        m = v_max / (sp.sqrt(p) + 1)
        k = -MIN_ACC / m / m
        return k * (self.speed - m) * (self.speed - m) + MIN_ACC

    # 单纯计算前车和自己的距离，不含车长
    def __calc_pure_interDistance(self, previous):
        if (not previous):
            return ROAD_LENGTH - self.location[0, 1]
        # assert  previous.__class__
        return previous.location[0, 1] - self.location[0, 1] - self.length - previous.length

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
        tem_a = -(epsilon + lam_para + sigma) / T
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

    def follow_car_for_platoon(self, STRATEGY, precious):
        temp_a = 0.0
        if (not precious):
            # 如果前车为空，说明自己是leader
            if (self.speed <= TURN_MAX_V):
                temp_a = self.__engine_speed_up_acc_curve(p=0.3)
            elif (self.speed > MAX_V):
                delta_v = sp.abs(self.speed - MAX_V)
                temp_a = -self.__engine_speed_up_acc_curve(self.speed - delta_v) * 0.5
        else:
            v1 = self.speed  # 自己的速度
            v2 = precious.speed  # 前车的速度
            if (precious.acc < 0.0):
                v2 += AI_DT * precious.acc
            v1 = v1 if v1 > 0 else 0.0
            v2 = v2 if v2 > 0 else 0.0
            s = self.__calc_pure_interDistance(precious)

            # 根据策略选择跟驰的方式
            if (STRATEGY == 'ACC'):
                self.__follow_car_ACC(s, precious)
            elif (STRATEGY == 'CACC'):
                if (not self.leader) or (self.id == self.leader.id):
                    self.__follow_car_ACC(s, precious)  # 调用ACC来补救
                else:
                    self.__follow_car_CACC(s, precious)  # 调用正宗的CACC


