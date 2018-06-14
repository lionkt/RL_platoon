"""
Environment is a Robot Arm. The arm tries to get to the blue point.
The environment will return a geographic (distance) information for the arm to learn.

The far away from blue point the less reward; touch blue r+=1; stop at blue for a while then get r=+10.

You can train this RL by using LOAD = False, after training, this model will be store in the a local folder.
Using LOAD = True to reload the trained model for playing.

You can customize this script in a way you want.

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/

Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
tensorflow >= 1.0.1
"""

import tensorflow as tf
import numpy as np
import os
import shutil
import car_env_DDPG_3cars as car_env
import plot_funcion as my_plot
import plot_train as train_plot
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# np.random.seed(1)
# tf.set_random_seed(1)

MAX_EPISODES = 900  # 1200
# MAX_EP_STEPS = 200
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-3  # learning rate for critic
GAMMA = 0.99  # reward discount， original 0.999
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128     # 32 get better output than 16
VAR_MIN = 0.005       # 0.05


# LOAD = False
LOAD = True
OUTPUT_GRAPH = True

# USE_RL_METHOD = False    # 判断是用传统的跟驰控制，还是用RL控制
USE_RL_METHOD = True    # 判断是用传统的跟驰控制，还是用RL控制
INIT_CAR_DISTANCE = car_env.INIT_CAR_DISTANCE  # 初始时车辆的间隔


output_file_name = './OutputImg/singe_decision_calculateTime.txt'
output_file = open(output_file_name,'w')


STATE_DIM = car_env.STATE_DIM
ACTION_DIM = car_env.ACTION_DIM
ACTION_BOUND = car_env.ACTION_BOUND

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('A'):
    A = tf.placeholder(tf.float32, shape=[None, ACTION_DIM], name='a')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l1', trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l3', trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w, name='a',
                                          trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound,
                                       name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s, a):  # batch update
        self.sess.run(self.train_op, feed_dict={S: s, A: a})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]  # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(
                -self.lr / BATCH_SIZE)  # (- learning rate) for ascent policy, div to take mean
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, t_replace_iter, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.q = self._build_net(S, A, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net',
                                      trainable=False)  # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, A)[0]  # tensor of gradients of each sample (None, a_dim)

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.01)

            with tf.variable_scope('l1'):
                n_l1 = 200
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l3', trainable=trainable)
            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b,
                                    trainable=trainable)  # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, A: a, R: r, S_: s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
        self.t_replace_counter += 1


class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]


config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.4  # 至少一块卡上留70%的显存，保证5个进程能跑起来
sess = tf.Session()

# Create actor and critic for 2-car following.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a_)
actor.add_grad_to_graph(critic.a_grads)
M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

# Create actor and critic for 3-car following.
# actor_new = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
# critic_new = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor_new.a_)
# actor_new.add_grad_to_graph(critic_new.a_grads)
# M_new = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)


saver = tf.train.Saver()
path = './' + 'Data/3_cars_following/'

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter('DDPG_logs/3_cars_following/', graph=sess.graph)


def train():
    # record field
    reward_list = []
    explore_list = []
    info_list = []
    observation_list = []
    plot_interval = 8   # 绘制训练图像的次数
    plot_iter = 1       # 当前的训练图绘制次数
    # train parameters
    var = 5  # control exploration, original 2.5
    var_damp = 0.999958  # var damping ratio, original 0.99995
    last_a = 0  # 上一个加速度值
    Carlist = []
    for ep in range(MAX_EPISODES):

        # 每个episode都要reset一下
        Carlist.clear()
        time_tag = 0.0
        car1 = car_env.car(id=0, role='leader', ingaged_in_platoon=False,
                           tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6, location=[0, INIT_CAR_DISTANCE*2])
        car2 = car_env.car(id=1, role='follower', ingaged_in_platoon=False,
                           tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6, location=[0, INIT_CAR_DISTANCE])
        car3 = car_env.car(id=2, role='follower', ingaged_in_platoon=False,
                           tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6, location=[0, 0])
        # 将新车加入车队
        if len(Carlist) == 0:
            Carlist.append(car1)
            Carlist.append(car2)
            Carlist.append(car3)
        # 设置参与车队的车辆，根据build_platoon，更新是否加入platoon的标志位
        CarList_update_platoon_info(Carlist, des_platoon_size=3, build_platoon=True)
        s = car_env.reset(Carlist)
        ep_reward = 0

        while True:
            # while True:
            # 时间戳更新
            time_tag += car_env.AI_DT

            # Added exploration noise
            a = actor.choose_action(s)
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)  # add randomness to action selection for exploration
            s_, done, info = car_env.step_next(Carlist, time_tag, action=a)
            r = car_env.get_reward_function(s_, (Carlist[2].acc-last_a)/car_env.AI_DT)
            # r = car_env.get_reward_table(s_)

            last_a = Carlist[2].acc  # 旧加速度更新
            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY:
                var = max([var * var_damp, VAR_MIN])  # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s, b_a)

            s = s_
            ep_reward += r

            if done:
                # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep, result, '| R: %i' % int(ep_reward), '| Explore: %.2f' % var, '| info: ', info,
                      '| dist-err(f1-f2):%.2f' % s[1],'| speed-err(f1-f2):%.2f' % s[0],'| speed-err(le-f2):%.2f' % s[2])
                ## save data for plot
                reward_list.append(int(ep_reward))
                explore_list.append(var)
                info_list.append(info)
                observation_list.append(s)
                break
        # 画一下最后一次的图像
        if ep == MAX_EPISODES - 1:
            train_plot.plot_train_core(reward_list, explore_list, info_list, observation_list,
                                       write_flag=False, title_in=1*100)
            my_plot.plot_data(Carlist, write_flag=True)
        # 画一下训练过程中的图像
        if ep == MAX_EPISODES//plot_interval*plot_iter:
            plot_iter += 1
            train_plot.plot_train_core(reward_list, explore_list, info_list, observation_list,
                                       write_flag=False, title_in=ep/MAX_EPISODES*100)

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./' + 'Data/3_cars_following/', 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=True)
    print("\nSave Model %s\n" % save_path)


def eval():
    Carlist = []
    # 每个episode都要reset一下
    Carlist.clear()
    time_tag = 0.0
    temp_init_speed = 0.0 / 3.6
    car1 = car_env.car(id=0, role='leader', ingaged_in_platoon=True, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, init_speed=temp_init_speed, location=[0, INIT_CAR_DISTANCE*3])
    car2 = car_env.car(id=1, role='follower', ingaged_in_platoon=True, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, init_speed=temp_init_speed, location=[0, INIT_CAR_DISTANCE*2])
    car3 = car_env.car(id=2, role='follower', ingaged_in_platoon=True, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, init_speed=temp_init_speed, location=[0, INIT_CAR_DISTANCE])
    car4 = car_env.car(id=3, role='follower', ingaged_in_platoon=True, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, init_speed=temp_init_speed, location=[0, 0])
    car5 = car_env.car(id=4, role='follower', ingaged_in_platoon=True, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, init_speed=temp_init_speed, location=[0, -INIT_CAR_DISTANCE])
    # car6 = car_env.car(id=5, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
    #                    tar_speed=60.0 / 3.6, init_speed=temp_init_speed, location=[0, -INIT_CAR_DISTANCE*2])
    # 将新车加入车队
    if len(Carlist) == 0:
        Carlist.append(car1)
        Carlist.append(car2)
        Carlist.append(car3)
        Carlist.append(car4)
        Carlist.append(car5)
        # Carlist.append(car6)
    # CarList_update_platoon_info(Carlist, des_platoon_size=len(Carlist), build_platoon=True)  # 把车辆加入车队

    s = car_env.reset(Carlist)
    done = False
    global output_file
    while True:
        # 时间戳更新
        time_tag += car_env.AI_DT

        t_start = time.time()

        # 多车同时加入仿真的计算
        Carlist[0].calculate(Carlist[0], STRATEGY='ACC', time_tag=time_tag, action=None)  # 先算头车
        Carlist[1].calculate(Carlist[0:2], STRATEGY='ACC', time_tag=time_tag, action=None)  # 先算第二辆
        for car_index in range(len(Carlist)):
            if car_index <= 1:
                continue
            if car_index == 2:
                temp_list = []  # 3辆车的数组
                temp_list.append(Carlist[car_index - 2])
                temp_list.append(Carlist[car_index - 1])
                temp_list.append(Carlist[car_index])
            elif car_index >= 3:
                temp_list = []  # 3辆车的数组
                temp_list.append(Carlist[0])
                temp_list.append(Carlist[car_index - 1])
                temp_list.append(Carlist[car_index])
            s, done, info = car_env.get_obs_done_info(temp_list, time_tag)  # 先读取一下当前的状态
            a = actor.choose_action(s)  # 根据当前状态，从训练好的网络选择动作
            temp_list[2].calculate(temp_list, STRATEGY='RL', time_tag=time_tag, action=a)  # 将输入的动作用于运算
            s_, done, info = car_env.get_obs_done_info(temp_list, time_tag)  # 更新一下当前的状态

        # 信息更新
        turns = 0
        while turns <= car_env.AI_DT:
            car_env.CarList_update_info_core(Carlist, car_env.UPDATE_TIME_PER_DIDA)
            turns += car_env.UPDATE_TIME_PER_DIDA

        t_end = time.time()


        # 输出计算的时间
        oneline = '%f' % time_tag + ',%f' % (t_end - t_start)
        output_file.write(oneline + '\n')

        # 判断仿真是否结束
        if done:
            output_file.close()
            break

    my_plot.plot_data(Carlist, write_flag=True)


# 根据build_platoon，更新是否加入platoon的信息
def CarList_update_platoon_info(Carlist, des_platoon_size, build_platoon):
    if build_platoon == False:
        for single_car in Carlist:
            single_car.engaged_in_platoon = False
    else:
        for single_car in Carlist:
            single_car.leader = Carlist[0]
        assert len(Carlist) >= des_platoon_size, '期望长度大于CarList总长度'

        for single_car in Carlist:
            single_car.engaged_in_platoon = True


# 采用两种或以上的跟驰策略进行组合
def multi_strategy_eval():
    Carlist = []
    # 每个episode都要reset一下
    Carlist.clear()
    time_tag = 0.0
    car1 = car_env.car(id=0, role='leader', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, 75])
    car2 = car_env.car(id=1, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, 50])
    car3 = car_env.car(id=2, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, 25])
    car4 = car_env.car(id=3, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, 0])
    # 将新车加入车队
    if len(Carlist) == 0:
        Carlist.append(car1)
        Carlist.append(car2)
        Carlist.append(car3)
        Carlist.append(car4)

    s = car_env.reset(Carlist)
    while True:
        # 时间戳更新
        time_tag += car_env.AI_DT

        # 多车同时加入仿真的计算
        done = False
        CarList_update_platoon_info(Carlist, des_platoon_size=4, build_platoon=True)  # 组建车队
        Carlist[0].calculate(Carlist[0], STRATEGY='RL', time_tag=time_tag, action=None)  # 先算头车，头车走自己的路，不受RL控制
        for pre_car_index in range(len(Carlist) - 1):
            temp_list = []  # 只存了两辆车的数组，专门针对RL训练出的双车跟驰模型
            temp_list.append(Carlist[pre_car_index])
            temp_list.append(Carlist[pre_car_index + 1])
            s, done, info = car_env.get_obs_done_info(temp_list, time_tag)  # 先读取一下当前的状态
            a = actor.choose_action(s)  # 根据当前状态，从训练好的网络选择动作

            # temp_list[1].calculate(Carlist, STRATEGY='MULTI', time_tag=time_tag, action=a)  # 将输入的动作用于运算
            temp_list[1].calculate(Carlist, STRATEGY='RL', time_tag=time_tag, action=a)  # 将输入的动作用于运算
            s_, done, info = car_env.get_obs_done_info(temp_list, time_tag)  # 更新一下当前的状态

        # 信息更新
        turns = 0
        while turns <= car_env.AI_DT:
            car_env.CarList_update_info_core(Carlist, car_env.UPDATE_TIME_PER_DIDA)
            turns += car_env.UPDATE_TIME_PER_DIDA

        # 判断仿真是否结束
        if done:
            break

    my_plot.plot_data(Carlist, write_flag=True)


def conventional_follow(STRATRGY):
    Carlist = []
    # 每个episode都要reset一下
    Carlist.clear()
    time_tag = 0.0
    temp_init_speed = 0.0 / 3.6
    car1 = car_env.car(id=0, role='leader', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, init_speed=temp_init_speed, location=[0, 75])
    car2 = car_env.car(id=1, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, init_speed=temp_init_speed, location=[0, 50])
    car3 = car_env.car(id=2, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, init_speed=temp_init_speed, location=[0, 25])
    car4 = car_env.car(id=3, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, init_speed=temp_init_speed, location=[0, 0])
    car5 = car_env.car(id=4, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, init_speed=temp_init_speed, location=[0, -25])
    # 将新车加入车队
    if len(Carlist) == 0:
        Carlist.append(car1)
        Carlist.append(car2)
        Carlist.append(car3)
        Carlist.append(car4)
        Carlist.append(car5)

    CarList_update_platoon_info(Carlist, des_platoon_size=5, build_platoon=True)

    global output_file
    while True:
        # 时间戳更新
        time_tag += car_env.AI_DT

        t_start = time.time()

        # 多车同时加入仿真的计算
        done = False
        car_env.CarList_calculate(Carlist, STRATEGY=STRATRGY, time_tag=time_tag, action=None)
        s_, done, info = car_env.get_obs_done_info(Carlist, time_tag)  # 更新一下当前的状态

        # 信息更新
        turns = 0
        while turns <= car_env.AI_DT:
            car_env.CarList_update_info_core(Carlist, car_env.UPDATE_TIME_PER_DIDA)
            turns += car_env.UPDATE_TIME_PER_DIDA

        t_end = time.time()
        # 输出计算的时间
        oneline = '%f' % time_tag + ',%f' % (t_end - t_start)
        output_file.write(oneline + '\n')

        # 判断仿真是否结束
        if done:
            output_file.close()
            break

    my_plot.plot_data(Carlist,write_flag=True)



if __name__ == '__main__':
    if USE_RL_METHOD:
        if LOAD:
            eval()
        else:
            train()
    else:
        conventional_follow('ACC')




# if __name__ == '__main__':
#     Carlist = []
#     # 每个episode都要reset一下
#     Carlist.clear()
#     time_tag = 0.0
#     car1 = car_env.car(id=0, role='leader', ingaged_in_platoon=False,
#                        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6, location=[0, 75])
#     car2 = car_env.car(id=1, role='follower', ingaged_in_platoon=False,
#                        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6, location=[0, 50])
#     car3 = car_env.car(id=2, role='follower', ingaged_in_platoon=False,
#                        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6, location=[0, 25])
#     car4 = car_env.car(id=3, role='follower', ingaged_in_platoon=False,
#                        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6, location=[0, 0])
#     # 将新车加入车队
#     if len(Carlist) == 0:
#         Carlist.append(car1)
#         Carlist.append(car2)
#         # Carlist.append(car3)
#         # Carlist.append(car4)
#
#     CarList_update_platoon_info(Carlist, des_platoon_size=2, build_platoon=True)
#     s = car_env.reset(Carlist)
#     while True:
#         # 时间戳更新
#         time_tag += car_env.AI_DT
#         # 多车同时加入仿真的计算
#         done = False
#
#         s, done, info = car_env.step_next(Carlist, time_tag, action=None)
#
#         # 判断仿真是否结束
#         if done:
#             break
#
#     my_plot.plot_data(Carlist)