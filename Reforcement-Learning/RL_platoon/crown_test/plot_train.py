import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import car_env as car_env
import numpy as np


# 绘制train过程的函数
def plot_train_core(reward_list, explore_list, info_list, observation_list, write_flag):
    observation_list = np.array(observation_list)
    plt.figure('train')
    plt.subplot(211)
    plt.plot(reward_list, linewidth=2)
    plt.title('train reward')
    plt.subplot(212)
    plt.plot(explore_list)
    plt.title('train explore value')
    out_png = './OutputImg/train.png'  # save file
    plt.savefig(out_png, dpi=300)

    #####
    plt.figure('l-f1-f2 parameters')
    plt.subplot(311)
    plt.plot(observation_list[:,1], linewidth=2)
    length_inter_space = list(np.arange(0, len(observation_list), 1))
    desired_inter_space = list(car_env.DES_PLATOON_INTER_DISTANCE * np.ones(len(length_inter_space)))
    plt.plot(length_inter_space, desired_inter_space, '--', linewidth=1.5, label='desired inter-space')
    plt.ylim(-20,40)
    plt.grid(True)
    plt.title('f1-f2 distance')

    plt.subplot(312)
    plt.plot(observation_list[:, 0], linewidth=2)
    plt.grid(True)
    plt.title('f1-f2 speed error')

    plt.subplot(313)
    plt.plot(observation_list[:, 2], linewidth=2)
    plt.title('leader-f2 speed error')
    plt.grid(True)
    out_png = './OutputImg/l-f1-f2 parameters.png'  # save file
    plt.savefig(out_png, dpi=300)

    plt.show()




