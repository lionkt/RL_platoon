import matplotlib.pyplot as plt
import car_env_DDPG_3cars as car_env
import numpy as np

DES_PLATOON_INTER_DISTANCE = car_env.DES_PLATOON_INTER_DISTANCE + 5
MAX_pure_distance = 25
MAX_pure_v = 3.5


def get_reward1(pure_interDistance):
    # 关于距离的reward
    if pure_interDistance <= DES_PLATOON_INTER_DISTANCE:
        r1 = 3 / (np.abs(pure_interDistance - DES_PLATOON_INTER_DISTANCE) + 0.008) - 3 / (pure_interDistance/4 + 0.005)
    elif pure_interDistance <= MAX_pure_distance:
        r1 = 3 / (np.abs(pure_interDistance - DES_PLATOON_INTER_DISTANCE) + 0.01) - 1 / (
            np.abs(pure_interDistance - MAX_pure_distance) + 0.03)
    else:
        r1 = 1 / (np.abs(MAX_pure_distance - DES_PLATOON_INTER_DISTANCE) + 0.05) - 1 / (
            np.abs(MAX_pure_distance - MAX_pure_distance) + 0.04)

    return r1


def get_reward2(delta_v_f2_with_f1):
    # 关于第二辆车速度的reward
    if delta_v_f2_with_f1 <= -MAX_pure_v:
        r2 = 1 / (np.abs(-MAX_pure_v - 0.0) + 0.05) - 1 / (np.abs(-MAX_pure_v + MAX_pure_v) + 0.04)
    elif delta_v_f2_with_f1 <= 0.0:
        r2 = 1 / (np.abs(delta_v_f2_with_f1 - 0.0) + 0.05) - 1 / (np.abs(delta_v_f2_with_f1 + MAX_pure_v) + 0.04)
    elif delta_v_f2_with_f1 <= MAX_pure_v:
        r2 = 1 / (np.abs(delta_v_f2_with_f1 - 0.0) + 0.05) - 1 / (np.abs(delta_v_f2_with_f1 - MAX_pure_v) + 0.04)
    else:
        r2 = 1 / (np.abs(MAX_pure_v - 0.0) + 0.03) - 1 / (np.abs(MAX_pure_v - MAX_pure_v) + 0.04)
    # return r1 * 0.123 + r2 * 0.045
    # return r1 * 0.053 + r2 * 0.045
    return r2



def get_reward3(post_jerk):
    turn_point1 = [1.0, 0]
    turn_point2 = [3.5, -2]
    turn_point3 = [5, -7]
    if abs(post_jerk) <= turn_point1[0]:
        r3 = turn_point1[1] / turn_point1[0] * abs(post_jerk)
    elif abs(post_jerk) <= turn_point2[0]:
        r3 = (turn_point2[1] - turn_point1[1]) / (turn_point2[0] - turn_point1[0]) * (abs(post_jerk) - turn_point1[0]) + \
             turn_point1[1]
    else:
        temp_r3 = (turn_point3[1] - turn_point2[1]) / (turn_point3[0] - turn_point2[0]) * (
                    abs(post_jerk) - turn_point2[0]) + turn_point2[1]
        # if abs(pure_interDistance) <= 0.8 * MAX_pure_distance:
        #     r3 = temp_r3
        # else:
        #     r3 = temp_r3 / 2
        r3 = temp_r3
    return r3

height = list(np.arange(-80, 30, 1))
desired_distance_list = list(DES_PLATOON_INTER_DISTANCE * np.ones(len(height)))
height_v =list(np.arange(-25, 20, 1))
desired_v_list = list(0.0 * np.ones(len(height_v)))

pure_distance_l = list(np.arange(0, DES_PLATOON_INTER_DISTANCE, 0.001))
pure_distance_h = list(np.arange(DES_PLATOON_INTER_DISTANCE, MAX_pure_distance, 0.001))
delta_v_l = list(np.arange(-MAX_pure_v, 0.0, 0.001))
delta_v_h = list(np.arange(0.0, MAX_pure_v, 0.001))
post_jerk = list(np.arange(-5, 5, 0.001))

r1_l = []
r1_h = []
r2_l = []
r2_h = []
r3 = []

for index in range(len(pure_distance_l)):
    r1_l.append(get_reward1(pure_interDistance=pure_distance_l[index]))
for index in range(len(pure_distance_h)):
    r1_h.append(get_reward1(pure_interDistance=pure_distance_h[index]))
for index in range(len(delta_v_l)):
    r2_l.append(get_reward2(delta_v_f2_with_f1=delta_v_l[index]))
for index in range(len(delta_v_h)):
    r2_h.append(get_reward2(delta_v_f2_with_f1=delta_v_h[index]))
for i in range(len(post_jerk)):
    r3.append(get_reward3(post_jerk=post_jerk[i]))

plt.figure()
plt.plot(pure_distance_l, r1_l, linewidth=2, label='less than desired-spacing')
plt.plot(pure_distance_h, r1_h, color='red', linewidth=2, label='more than desired-spacing')
plt.plot(desired_distance_list,height,'--',linewidth=1.5,label='desired-spacing')
plt.ylim(-80, 40)
plt.xlabel('inter-spacing (m)')
plt.ylabel('reward value')
plt.legend(loc=4)
plt.grid()
# plt.show()

plt.figure()
plt.plot(delta_v_l, r2_l, linewidth=2, label='less than desired-$\Delta$speed')
plt.plot(delta_v_h, r2_h, color='red', linewidth=2, label='more than desired-$\Delta$speed')
plt.plot(desired_v_list,height_v,'--',linewidth=1.5,label='desired-$\Delta$speed')
plt.xlabel('$\Delta$speed (m/s)')
plt.ylabel('reward value')
plt.legend(loc=4)
plt.grid()
# plt.show()

plt.figure()
plt.plot(post_jerk,r3, linewidth=2)
plt.xlabel('Jerk (m/$s^3$)')
plt.ylabel('reward value')
plt.grid()
plt.show()