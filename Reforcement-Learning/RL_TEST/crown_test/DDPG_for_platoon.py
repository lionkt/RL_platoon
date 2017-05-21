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
import car_env_DDPG as car_env
import plot_funcion as my_plot

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 50
# MAX_EP_STEPS = 200
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.999  # reward discount
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 16
VAR_MIN = 0.1
LOAD = True
n_model = 1

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
            net = tf.layers.dense(s, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
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
            opt = tf.train.RMSPropOptimizer(
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
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

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
            net = tf.layers.dense(net, 200, activation=tf.nn.relu6,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l3',
                                  trainable=trainable)
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


sess = tf.Session()

# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A)
critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = './' + 'Data'

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())


def train():
    var = 2.  # control exploration
    Carlist = []
    for ep in range(MAX_EPISODES):

        # 每个episode都要reset一下
        Carlist.clear()
        time_tag = 0.0
        car1 = car_env.car(
            id=0,
            role='leader',
            ingaged_in_platoon=False,
            tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
            tar_speed=60.0 / 3.6,
            location=[0, 50]
        )
        car2 = car_env.car(
            id=1,
            role='follower',
            ingaged_in_platoon=False,
            tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
            tar_speed=60.0 / 3.6,
            location=[0, 25]
        )
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
                actor.learn(b_s, b_a)

            s = s_
            ep_reward += r

            if done:
                # if done:
                result = '| done' if done else '| ----'
                print('Ep:', ep,
                      result,
                      '| R: %i' % int(ep_reward),
                      '| Explore: %.2f' % var,
                      '| info: ', info,
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


def eval():
    Carlist = []
    # 每个episode都要reset一下
    Carlist.clear()
    time_tag = 0.0
    car1 = car_env.car(
        id=0,
        role='leader',
        ingaged_in_platoon=False,
        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
        tar_speed=60.0 / 3.6,
        location=[0, 50]
    )
    car2 = car_env.car(
        id=1,
        role='follower',
        ingaged_in_platoon=False,
        tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
        tar_speed=60.0 / 3.6,
        location=[0, 25]
    )
    # 将新车加入车队
    if len(Carlist) == 0:
        Carlist.append(car1)
        Carlist.append(car2)
        # 设置参与车队的车辆，根据build_platoon，更新是否加入platoon的标志位
    s = car_env.reset(Carlist)
    while True:
        # 时间戳更新
        time_tag += car_env.AI_DT

        a = actor.choose_action(s)
        s_, done, info = car_env.step_next(Carlist, time_tag, action=a)
        r = car_env.get_reward_function(s_)
        s = s_
        if done:
            break
    my_plot.plot_data(Carlist)


if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        train()
