"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
import os
import shutil
import car_env_DDPG as car_env
import plot_funcion as my_plot

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

np.random.seed(1)
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 150
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.995     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)][1]            # you can try different target replacement strategies
MEMORY_CAPACITY = 15000
BATCH_SIZE = 40
VAR_MIN = 0.1

LOAD = True
OUTPUT_GRAPH = True

# parameters from car environment
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

###############################  Actor  ####################################

class Actor(object):
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.01)
            net = tf.layers.dense(s, 100, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 100, activation=tf.nn.relu6, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l3', trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # self.a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)
            # morvandemo用的是1层，扩展到了3层
            with tf.variable_scope('l1'):
                n_l1 = 100
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 100, activation=tf.nn.relu6, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, kernel_initializer=init_w, bias_initializer=init_b,
                                  name='l3', trainable=trainable)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory  ####################

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


#####################  build actor and critic  ####################
sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACEMENT)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = './Data/'

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("./DDPG_logs/new_DDPG/", sess.graph)


#####################  train  ####################
# def train_demo():
#     var = 3  # control exploration
#     for i in range(MAX_EPISODES):
#         s = env.reset()
#         ep_reward = 0
#
#         for j in range(MAX_EP_STEPS):
#
#             # Add exploration noise
#             a = actor.choose_action(s)
#             a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
#             s_, r, done, info = env.step(a)
#
#             M.store_transition(s, a, r / 10, s_)
#
#             if M.pointer > MEMORY_CAPACITY:
#                 var *= .9995    # decay the action randomness
#                 b_M = M.sample(BATCH_SIZE)
#                 b_s = b_M[:, :state_dim]
#                 b_a = b_M[:, state_dim: state_dim + action_dim]
#                 b_r = b_M[:, -state_dim - 1: -state_dim]
#                 b_s_ = b_M[:, -state_dim:]
#
#                 critic.learn(b_s, b_a, b_r, b_s_)
#                 actor.learn(b_s)
#
#             s = s_
#             ep_reward += r
#
#             if j == MAX_EP_STEPS-1:
#                 print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
#                 if ep_reward > -300:
#                     RENDER = True
#                 break


def train():
    var = 3  # control exploration
    Carlist = []
    for ep in range(MAX_EPISODES):
        # 每个episode都要reset一下
        Carlist.clear()
        time_tag = 0.0
        car1 = car_env.car(id=0, role='leader', ingaged_in_platoon=False,
                           tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6, location=[0, 50])
        car2 = car_env.car(id=1, role='follower', ingaged_in_platoon=False,
                           tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6, location=[0, 25])
        car3 = car_env.car(id=1, role='follower', ingaged_in_platoon=False,
                           tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE, tar_speed=60.0 / 3.6, location=[0, 0])
        # 将新车加入车队
        if len(Carlist) == 0:
            Carlist.append(car1)
            Carlist.append(car2)
        # 设置参与车队的车辆，根据build_platoon，更新是否加入platoon的标志位
        car_env.CarList_update_platoon_info(Carlist, des_platoon_size=2, build_platoon=True)
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
            r = car_env.get_reward_function(s_)
            # r = car_env.get_reward_table(s_)

            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY:
                var = max([var * .99995, VAR_MIN])  # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

                critic.learn(b_s, b_a, b_r, b_s_)
                actor.learn(b_s)

            s = s_
            ep_reward += r

            if done:
                # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep, result, '| R: %i' % int(ep_reward), '| Explore: %.2f' % var, '| info: ', info,
                      '| pure-dis:%.2f' % s[1])
                break
        # 画一下最后一次的图像
        if ep == MAX_EPISODES - 1:
            my_plot.plot_data(Carlist)

    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)
    ckpt_path = os.path.join('./' + 'Data', 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=True)
    print("\nSave Model %s\n" % save_path)


#####################  evaluate  ####################
def eval():
    Carlist = []
    # 每个episode都要reset一下
    Carlist.clear()
    time_tag = 0.0
    car1 = car_env.car(id=0, role='leader', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, 50])
    car2 = car_env.car(id=1, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, 25])
    car3 = car_env.car(id=2, role='follower', ingaged_in_platoon=False, tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                       tar_speed=60.0 / 3.6, location=[0, 0])
    # 将新车加入车队
    if len(Carlist) == 0:
        Carlist.append(car1)
        Carlist.append(car2)
        Carlist.append(car3)

    s = car_env.reset(Carlist)
    while True:
        # 时间戳更新
        time_tag += car_env.AI_DT

        # if len(Carlist) == 1 and time_tag >= 2:
        #     Carlist.append(car2)

        # 原始的两车计算
        # a = actor.choose_action(s)
        # s_, done, info = car_env.step_next(Carlist, time_tag, action=a)
        # s = s_
        # if done:
        #     break

        # 多车同时加入仿真的计算
        done = False
        Carlist[0].calculate(Carlist[0], STRATEGY='RL', time_tag=time_tag, action=None)  # 先算头车
        for pre_car_index in range(len(Carlist) - 1):
            temp_list = []  # 只存了两辆车的数组
            temp_list.append(Carlist[pre_car_index])
            temp_list.append(Carlist[pre_car_index + 1])
            s, done, info = car_env.get_obs_done_info(temp_list, time_tag)  # 先读取一下当前的状态
            a = actor.choose_action(s)  # 根据当前状态，从训练好的网络选择动作
            temp_list[1].calculate(temp_list, STRATEGY='RL', time_tag=time_tag, action=a)  # 将输入的动作用于运算
            s_, done, info = car_env.get_obs_done_info(temp_list, time_tag)  # 更新一下当前的状态

        # 信息更新
        turns = 0
        while turns <= car_env.AI_DT:
            car_env.CarList_update_info_core(Carlist, car_env.UPDATA_TIME_PER_DIDA)
            turns += car_env.UPDATA_TIME_PER_DIDA

        # 判断仿真是否结束
        if done:
            break

    my_plot.plot_data(Carlist)

# 根据build_platoon，更新是否加入platoon的信息
def CarList_update_platoon_info(Carlist, des_platoon_size, build_platoon):
    if build_platoon == False:
        for single_car in Carlist:
            single_car.engaged_in_platoon = False
    else:
        for single_car in Carlist:
            single_car.leader = Carlist[0]
        if len(Carlist) < des_platoon_size:
            for single_car in Carlist:
                single_car.engaged_in_platoon = False
        else:
            for single_car in Carlist:
                single_car.engaged_in_platoon = True



#####################  main function  ####################
if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        train()

