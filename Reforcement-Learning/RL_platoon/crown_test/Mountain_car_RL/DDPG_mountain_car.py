"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.

Using:
tensorflow 1.0
gym 0.8.0
"""

import tensorflow as tf
import numpy as np
# import gym
import os
import Mountain_car_RL.mountain_car_env as mountain_car_env
import Mountain_car_RL.Evaluate_func as eval_module

#####################  hyper parameters  ####################

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MAX_train_episode = 500
MAX_episode_length = 300
LR_A = 0.005    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.997     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

Eval_interval = 50
Eval_episode = 100

# ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'MountainCar-v0'
ENV_NAME = 'MountainCarContinuous-v0'
###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_actor(self.S, scope='eval', trainable=True)
            a_ = self._build_actor(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_critic(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_critic(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - TAU) * ta + TAU * ea), tf.assign(tc, (1 - TAU) * tc + TAU * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        probs =  self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]
        # act = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return probs

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_actor(self, s, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            net = tf.layers.dense(s, n_l1, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_critic(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################
if __name__ == '__main__':
    # env = gym.make(ENV_NAME)
    # env = env.unwrapped
    # env.seed(1)

    s_dim = mountain_car_env.NUM_FEATURE# env.observation_space.shape[0]
    # a_dim = env.action_space.n
    a_dim = 1   #env.action_space.shape[0]
    a_bound = mountain_car_env.ACT[1] # env.action_space.high

    ddpg = DDPG(a_dim, s_dim, a_bound)

    var = 5  # control exploration
    total_steps = 0
    avg_steps_list = []
    # begin main function
    for i_episode in range(MAX_train_episode):
        if (i_episode + 1) % Eval_interval == 0:
            print('=== Now finish %.3f' % ((i_episode + 1) / MAX_train_episode * 100), '% of ', str(MAX_train_episode),
                  'eps')

        # begin eval
        if (i_episode + 1) % Eval_interval == 0 or i_episode == 0:
            print('========== begin DDPG mountain car ==========')
            avg_steps = eval_module.eval_mountain_car(RL=ddpg, eval_eps=Eval_episode, reset_method=3,
                                                      reward_function=None)
            avg_steps_list.append(avg_steps)
            print('------ eval, avg steps: %.1f' % (avg_steps))


        # s = env.reset()
        s = mountain_car_env.random_reset(method=3)
        ep_reward = 0
        ep_step = 0
        while True:

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration

            # s_, r, done, info = env.step(a)
            s_, done = mountain_car_env.step_next(s, a)
            r = mountain_car_env.cal_reward(s_, reward_function=None)

            ddpg.store_transition(s, a, r, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                var *= .9995    # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            ep_step += 1
            if done or ep_step >= MAX_episode_length:
                break

    # output performance to file
    root_path = '../OutputImg/Mountain_car/'
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    output_file_name = 'DDPG' + '_MaxEp=' + str(MAX_train_episode) + '_MaxEpLen=' + str(
        MAX_episode_length) + '_AvgSteps.txt'
    write_buffer = np.array(avg_steps_list).transpose()
    np.savetxt(root_path + output_file_name, write_buffer)