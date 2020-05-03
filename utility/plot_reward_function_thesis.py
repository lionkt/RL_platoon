import matplotlib.pyplot as plt
import matplotlib
import car_env_DDPG_3cars as car_env
import numpy as np
import os

zhfont = matplotlib.font_manager.FontProperties(
    fname='C:\Windows\Fonts\simhei.ttf', size=11)
zhfont_legend = matplotlib.font_manager.FontProperties(
    fname='C:\Windows\Fonts\simhei.ttf', size=10)

CAR_LENGTH = car_env.CAR_LENGTH
DES_PLATOON_INTER_DISTANCE = car_env.DES_PLATOON_INTER_DISTANCE+CAR_LENGTH
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


def get_reward4(jerk):
    if np.abs(jerk) < 1:
        r4 = 0
    elif 1 <= np.abs(jerk) <= 3.5:
        r4 = -4*np.abs(jerk)/5 + 4/5
    else:
        r4 = -10*np.abs(jerk)/3 + 29/3
    return r4


height = list(np.arange(-80, 30, 1))
desired_distance_list = DES_PLATOON_INTER_DISTANCE * np.ones(len(height))
height_v =np.arange(-25, 20, 1)
desired_v_list = 0.0 * np.ones(len(height_v))

pure_distance_l = np.arange(0, DES_PLATOON_INTER_DISTANCE, 0.001)
pure_distance_h = np.arange(DES_PLATOON_INTER_DISTANCE, MAX_pure_distance, 0.001)
delta_v_l = np.arange(-MAX_pure_v, 0.0, 0.001)
delta_v_h = np.arange(0.0, MAX_pure_v, 0.001)
jerk = np.arange(-4.6,4.6,0.001)


r1_l = [get_reward1(x) for x in pure_distance_l]
r1_h = [get_reward1(x) for x in pure_distance_h]
r2_l = [get_reward2(x) for x in delta_v_l]
r2_h = [get_reward2(x) for x in delta_v_h]
r4 = [get_reward4(x) for x in jerk]


plt.figure(figsize=(5,3.5))
plt.plot(pure_distance_l, r1_l, linewidth=2)
plt.plot(pure_distance_h, r1_h, color='red', linewidth=2, )
plt.plot(desired_distance_list,height,'--',linewidth=1.5)
plt.legend(['小于期望间距$d_{des}$','大于期望间距$d_{des}$','期望间距$d_{des}$'],prop=zhfont_legend)
plt.ylim(-40, 35)
plt.xlabel('跟驰车间距离 d',fontproperties=zhfont)
plt.ylabel('奖励函数值 $r_1(d)$',fontproperties=zhfont)
plt.legend(loc=4)
plt.grid()
output_fig_path = './OutputImg/reward_function'
plt.savefig(os.path.join(output_fig_path,'reward_distance.png'),dpi=300)
# plt.show()

plt.figure(figsize=(5,3.5))
plt.plot(delta_v_l, r2_l, linewidth=2)
plt.plot(delta_v_h, r2_h, color='red', linewidth=2)
plt.plot(desired_v_list,height_v,'--',linewidth=1.5)
plt.legend(['小于期望速度差$\Delta v_{des}$','大于期望速度差$\Delta v_{des}$','期望速度差$\Delta v_{des}$'],
           prop=zhfont_legend)
plt.xlabel('跟驰速度偏差 $\Delta v$ (m/s)',fontproperties=zhfont)
plt.ylabel('奖励函数值 $r_2(\Delta v)$',fontproperties=zhfont)
plt.legend(loc=2)
plt.grid()
output_fig_path = './OutputImg/reward_function'
plt.savefig(os.path.join(output_fig_path,'reward_delta_v.png'),dpi=300)

plt.figure(figsize=(5,3.5))
plt.plot(jerk, r4)
plt.xlabel('急动度 jerk (m/$s^2$)',fontproperties=zhfont)
plt.ylabel('奖励函数值 $r_4(jerk)$',fontproperties=zhfont)
plt.legend(loc=2)
plt.grid()
output_fig_path = './OutputImg/reward_function'
plt.savefig(os.path.join(output_fig_path,'reward_jerk.png'),dpi=300)


plt.show()
