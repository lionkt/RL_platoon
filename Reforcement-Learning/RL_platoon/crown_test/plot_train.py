import matplotlib.pyplot as plt
import car_env_DDPG_3cars as car_env_3_car
import numpy as np
import os


# 绘制train过程的函数
def plot_train_core(reward_list, explore_list, info_list, observation_list, write_flag, title_in, output_path):
    # mkdir for data output
    full_path = output_path + './OutputImg/'
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    observation_list = np.array(observation_list)
    figure_name = 'train parameters, Now ' + str(title_in) + '%'
    plt.figure(figure_name)
    plt.subplot(211)
    plt.plot(reward_list, linewidth=2)
    plt.title('train reward ' + str(title_in) + '%')

    if len(explore_list) > 0:
        plt.subplot(212)
        plt.plot(explore_list)
        plt.title('train explore value ' + str(title_in) + '%')

    out_png = full_path + 'train parameters.png'  # save file
    plt.savefig(out_png, dpi=300)

    #####
    figure_name = 'train l-f1-f2 parameters, Now ' + str(title_in) + '%'
    plt.figure(figure_name)
    plt.subplot(311)
    plt.plot(observation_list[:, 1], linewidth=2)
    length_inter_space = list(np.arange(0, len(observation_list), 1))
    desired_inter_space = list(car_env_3_car.DES_PLATOON_INTER_DISTANCE * np.ones(len(length_inter_space)))
    plt.plot(length_inter_space, desired_inter_space, '--', linewidth=1.5, label='desired inter-space')
    plt.ylim(-10, 40)
    plt.grid(True)
    plt.title('f1-f2 distance ' + str(title_in) + '%')

    plt.subplot(312)
    plt.plot(observation_list[:, 0], linewidth=2)
    plt.grid(True)
    plt.title('f1-f2 speed error ' + str(title_in) + '%')

    plt.subplot(313)
    plt.plot(observation_list[:, 2], linewidth=2)
    plt.title('leader-f2 speed error ' + str(title_in) + '%')
    plt.grid(True)
    out_png = full_path + 'train l-f1-f2 parameters.png'  # save file
    plt.savefig(out_png, dpi=300)

    # plt.show()
