import matplotlib.pyplot as plt
import car_env as car_env
import numpy as np

DES_PLATOON_INTER_DISTANCE = car_env.DES_PLATOON_INTER_DISTANCE
MAX_pure_distance = 20
MAX_pure_v = 3.5


def get_reward1(pure_interDistance):
    if pure_interDistance <= DES_PLATOON_INTER_DISTANCE:
        r1 = 1 / (np.abs(pure_interDistance - DES_PLATOON_INTER_DISTANCE) + 0.02) - 3 / (pure_interDistance + 0.005)
    elif pure_interDistance <= MAX_pure_distance:
        r1 = 3 / (np.abs(pure_interDistance - DES_PLATOON_INTER_DISTANCE) + 0.02) - 1 / (
            np.abs(pure_interDistance - MAX_pure_distance) + 0.03)
    else:
        r1 = 1 / (np.abs(MAX_pure_distance - DES_PLATOON_INTER_DISTANCE) + 0.05) - 1 / (
            np.abs(MAX_pure_distance - MAX_pure_distance) + 0.04)

    return r1


def get_reward2(delta_v):
    if delta_v <= -MAX_pure_v:
        r2 = 1 / (np.abs(-MAX_pure_v - 0.0) + 0.05) - 1 / (np.abs(-MAX_pure_v + MAX_pure_v) + 0.04)
    elif delta_v <= 0.0:
        r2 = 1 / (np.abs(delta_v - 0.0) + 0.05) - 1 / (np.abs(delta_v + MAX_pure_v) + 0.04)
    elif delta_v <= MAX_pure_v:
        r2 = 1 / (np.abs(delta_v - 0.0) + 0.05) - 1 / (np.abs(delta_v - MAX_pure_v) + 0.04)
    else:
        r2 = 1 / (np.abs(MAX_pure_v - 0.0) + 0.03) - 1 / (np.abs(MAX_pure_v - MAX_pure_v) + 0.04)
    # return r1 * 0.123 + r2 * 0.045
    # return r1 * 0.053 + r2 * 0.045
    return r2


height = list(np.arange(-80, 30, 1))
desired_distance_list = list(DES_PLATOON_INTER_DISTANCE * np.ones(len(height)))
height_v =list(np.arange(-25, 20, 1))
desired_v_list = list(0.0 * np.ones(len(height_v)))

pure_distance_l = list(np.arange(0, DES_PLATOON_INTER_DISTANCE, 0.001))
pure_distance_h = list(np.arange(DES_PLATOON_INTER_DISTANCE, MAX_pure_distance, 0.001))
delta_v_l = list(np.arange(-MAX_pure_v, 0.0, 0.001))
delta_v_h = list(np.arange(0.0, MAX_pure_v, 0.001))

r1_l = []
r1_h = []
r2_l = []
r2_h = []

for index in range(len(pure_distance_l)):
    r1_l.append(get_reward1(pure_interDistance=pure_distance_l[index]))
for index in range(len(pure_distance_h)):
    r1_h.append(get_reward1(pure_interDistance=pure_distance_h[index]))
for index in range(len(delta_v_l)):
    r2_l.append(get_reward2(delta_v=delta_v_l[index]))
for index in range(len(delta_v_h)):
    r2_h.append(get_reward2(delta_v=delta_v_h[index]))

# plt.plot(pure_distance_l, r1_l, linewidth=2, label='less than desired-distance')
# plt.plot(pure_distance_h, r1_h, color='red', linewidth=2, label='more than desired-distance')
# plt.plot(desired_distance_list,height,'--',linewidth=1.5,label='desired-distance')
# plt.ylim(-80, 40)
# plt.xlabel('inter-distance/m')
# plt.ylabel('reward value')
# plt.legend(loc=4)
# plt.grid()
# plt.show()

plt.plot(delta_v_l, r2_l, linewidth=2, label='less than desired-delta_speed')
plt.plot(delta_v_h, r2_h, color='red', linewidth=2, label='more than desired-delta_speed')
plt.plot(desired_v_list,height_v,'--',linewidth=1.5,label='desired-delta_speed')
plt.xlabel('delta_speed/(m/s)')
plt.ylabel('reward value')
plt.legend(loc=2)
plt.grid()
plt.show()