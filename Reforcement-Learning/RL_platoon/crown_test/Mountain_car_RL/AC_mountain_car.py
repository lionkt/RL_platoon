"""
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.

Using:
tensorflow 1.0
gym 0.8.0
"""
# import gym
import numpy as np
import tensorflow as tf
import os
import Mountain_car_RL.mountain_car_env as mountain_car_env
import Mountain_car_RL.Evaluate_func as eval_module

np.random.seed(2)
tf.set_random_seed(2)  # reproducible


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Superparameters
OUTPUT_GRAPH = False
MAX_train_episode = 500
MAX_episode_length = 300
Eval_interval = 50
Eval_episode = 100

GAMMA = 0.995     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

# env = gym.make('MountainCar-v0')
# env.seed(1)  # reproducible
# env = env.unwrapped

n_features = mountain_car_env.NUM_FEATURE # env.observation_space.shape[0]
n_actions = mountain_car_env.NUM_ACT # env.action_space.n


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error


sess = tf.Session()

actor = Actor(sess, n_features=n_features, n_actions=n_actions, lr=LR_A)
critic = Critic(sess, n_features=n_features, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)


if __name__ == '__main__':
    total_steps = 0
    avg_steps_list = []
    # begin main function
    for i_episode in range(MAX_train_episode):
        if (i_episode + 1) % Eval_interval == 0:
            print('=== Now finish %.3f' % ((i_episode + 1) / MAX_train_episode * 100), '% of ', str(MAX_train_episode), 'eps')

        # begin eval
        if (i_episode + 1) % Eval_interval == 0 or i_episode == 0:
            print('========== begin Actor-critic mountain car ==========')
            avg_steps = eval_module.eval_mountain_car(RL=actor, eval_eps=Eval_episode, reset_method=3,
                                                      reward_function=None)
            avg_steps_list.append(avg_steps)
            print('------ eval, avg steps: %.1f' %(avg_steps))

        # begin train
        # s = env.reset()
        s = mountain_car_env.random_reset(method=3)
        ep_step = 0
        track_r = []
        while True:

            a = actor.choose_action(s)

            s_, done = mountain_car_env.step_next(s, a)
            r = mountain_car_env.cal_reward(s_, reward_function=None)

            track_r.append(r)

            td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
            if total_steps >= 1000:
                actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

            s = s_
            ep_step += 1
            total_steps += 1

            if done or ep_step >= MAX_episode_length:
                break

    root_path = '../OutputImg/Mountain_car/'
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    output_file_name = 'AC' + '_MaxEp=' + str(MAX_train_episode) + '_MaxEpLen=' + str(MAX_episode_length) + '_AvgSteps.txt'
    write_buffer = np.array(avg_steps_list).transpose()
    np.savetxt(root_path + output_file_name, write_buffer)