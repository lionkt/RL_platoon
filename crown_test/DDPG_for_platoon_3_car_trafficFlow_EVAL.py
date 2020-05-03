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
import time
import car_env_DDPG_3cars_trafficFlow_EVAL as car_env
import plot_funcion as my_plot
import plot_train as train_plot

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# np.random.seed(1)
# tf.set_random_seed(1)

MAX_EPISODES = 1200  # 1200
# MAX_EP_STEPS = 200
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-3  # learning rate for critic
GAMMA = 0.999  # reward discount， original 0.999
REPLACE_ITER_A = 1100
REPLACE_ITER_C = 1000
MEMORY_CAPACITY = 10000
BATCH_SIZE = 128     # 32 get better output than 16
VAR_MIN = 0.005       # 0.05


# STRATEGY_INPUT = 'RL'  # 跟驰所用的控制器
STRATEGY_INPUT = 'ACC'  # 跟驰所用的控制器

### 输出文件设定
if not os.path.exists('./OutputImg/trafficFlow/'):
    os.mkdir('./OutputImg/trafficFlow/')
output_file_name = './OutputImg/trafficFlow/change_lane_scenario_data.txt'
output_platoon_spacing_name = './OutputImg/trafficFlow/platoon_spacing.txt'
output_file = open(output_file_name, 'w')
output_platoon_spacing_file = open(output_platoon_spacing_name, 'w')
output_title_flag = False
output_platoon_title_flag = False
output_interval = 25 # 每隔output_interval输出一次

STATE_DIM = car_env.STATE_DIM
ACTION_DIM = car_env.ACTION_DIM
ACTION_BOUND = car_env.ACTION_BOUND

# all placeholder for tf
if STRATEGY_INPUT == 'RL':
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


# 在允许换道的场景下中对不同的跟驰控制器测试
def eval_trafficFlow(STRATEGY):
    # 控制常数
    PLATOON_GEN_TIME_INTERVAL = 20  # 间隔多少秒生成一个platoon
    left_max_v = car_env.left_max_v
    mid_max_v = car_env.mid_max_v
    right_max_v = car_env.right_max_v

    # 构建车道
    lane_left = car_env.lane(id=0, startx=car_env.LANE_WIDTH * 0.5, starty=0)
    lane_mid = car_env.lane(id=1, startx=car_env.LANE_WIDTH * (0.5 + 1), starty=0)
    lane_right = car_env.lane(id=2, startx=car_env.LANE_WIDTH * (0.5 + 2), starty=0)
    lane_list = [lane_left,lane_mid,lane_right]   # 3车道
    # lane_list = [lane_mid, lane_right]  # 2车道

    vehicle_list = []  # vehicle的总list
    CarList_list = []
    vehicle_valid_id = 0  # vehicle可以用的id起点

    time_tag = 0.0
    done = False
    finish_flag = False
    finish_info = ''
    last_gen_time_tag = 0.0
    while True:
        ###### 时间戳更新
        time_tag += car_env.AI_DT
        ###### 间隔到了，开始生产车队 ######
        if time_tag - last_gen_time_tag >= PLATOON_GEN_TIME_INTERVAL:
            # 检测能否生成platoon（避撞的考虑）
            min_location_y = car_env.ROAD_LENGTH
            for th_ in range(len(vehicle_list)):
                if vehicle_list[th_].location[1] < min_location_y:
                    min_location_y = vehicle_list[th_].location[1]
            # 如果最靠近起点的一个车，距离起点也有min_location_y
            if min_location_y >= car_env.INIT_CAR_DISTANCE:
                # 生成车道
                # lane_index = random.randint(0, len(lane_list) - 1)    # 随机在不同的车道生成（未完成）
                lane_index = len(lane_list)-1   # 只在最右边的车道生成，然后加速到车道允许的上限后，才能换道
                # 生成随机数作为platoon的长度
                mu = 3
                sigma = 1
                temp_car_num = np.random.normal(mu, sigma, 1)
                temp_car_num = np.round(temp_car_num)
                temp_car_num = temp_car_num if temp_car_num <= 5 else 5     # 至多产生5辆车
                temp_car_num = temp_car_num if temp_car_num >=1 else 1      # 至少产生1辆车
                temp_car_num = int(temp_car_num)
                # 根据生成的随机数构建车队
                temp_lane_max_v = 0.0
                if lane_index == 0:
                    temp_lane_max_v = left_max_v
                elif lane_index==1:
                    temp_lane_max_v=mid_max_v
                elif lane_index==2:
                    temp_lane_max_v=right_max_v
                else:
                    print("lane index error!!!!")
                    raise NameError
                temp_car_list = []
                for th_ in range(temp_car_num):
                    init_location_x = lane_list[lane_index].startx
                    if th_ == 0:
                        single_car = car_env.car(id=vehicle_valid_id, role='leader',
                                                 tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                                                 tar_speed=temp_lane_max_v, init_location_x=init_location_x,
                                                 init_speed=0.0, max_v=temp_lane_max_v, location=[0, 0],
                                                 ingaged_in_platoon=True, run_test=False, init_lane=lane_index)
                    else:
                        single_car = car_env.car(id=vehicle_valid_id, role='follower',
                                                 tar_interDis=car_env.DES_PLATOON_INTER_DISTANCE,
                                                 tar_speed=temp_lane_max_v, init_location_x=init_location_x,
                                                 init_speed=0.0, max_v=temp_lane_max_v,
                                                 location=[0, -car_env.INIT_CAR_DISTANCE * th_],
                                                 ingaged_in_platoon=True, run_test=False, init_lane=lane_index)
                    vehicle_valid_id +=1
                    vehicle_list.append(single_car)
                    temp_car_list.append(single_car)
                # 完成了一个车队的生成
                CarList_list.append(temp_car_list)
        ###### 开始运动学计算 ######
        for list_th in range(len(CarList_list)):
            if STRATEGY == 'RL':  # 使用RL方法
                CarList_list[list_th][0].calculate(CarList_list[list_th][0], STRATEGY='ACC', time_tag=time_tag,
                                                   action=None, vehicle_list=vehicle_list, lane_list=lane_list)  # 先算头车
                # CarList_list[list_th][1].calculate(CarList_list[list_th][0:2], STRATEGY='ACC', time_tag=time_tag,
                #                                    action=None, vehicle_list=vehicle_list, lane_list=lane_list)  # 先算第二辆
                for car_index in range(len(CarList_list[list_th])):
                    if car_index == 0:
                        continue
                    elif car_index == 1:
                        temp_list = []  # 2辆车的数组
                        temp_list.append(CarList_list[list_th][0])
                        temp_list.append(CarList_list[list_th][0])
                        temp_list.append(CarList_list[list_th][car_index])
                    elif car_index == 2:
                        temp_list = []  # 3辆车的数组
                        temp_list.append(CarList_list[list_th][car_index - 2])
                        temp_list.append(CarList_list[list_th][car_index - 1])
                        temp_list.append(CarList_list[list_th][car_index])
                    elif car_index >= 3:
                        temp_list = []  # 3辆车的数组
                        temp_list.append(CarList_list[list_th][0])
                        temp_list.append(CarList_list[list_th][car_index - 1])
                        temp_list.append(CarList_list[list_th][car_index])
                    s, done, info = car_env.get_obs_done_info(temp_list, time_tag)  # 先读取一下当前的状态
                    a = actor.choose_action(s)  # 根据当前状态，从训练好的网络选择动作
                    temp_list[2].calculate(temp_list, STRATEGY='RL', time_tag=time_tag, action=a,
                                           vehicle_list=vehicle_list, lane_list=lane_list)  # 将输入的动作用于运算
                    s_, done, info = car_env.get_obs_done_info(temp_list, time_tag)  # 更新一下当前的状态
            ### 不使用RL跟驰
            else:   # 不适用RL方法
                for j in range(len(CarList_list[list_th])):
                    CarList_list[list_th][j].calculate(CarList_list[list_th][0:j+1], STRATEGY=STRATEGY,
                                                       time_tag=time_tag, action=None,
                                                       vehicle_list=vehicle_list, lane_list=lane_list)  # 先算第二辆

            #### 信息更新
            turns = 0
            while turns <= car_env.AI_DT:
                car_env.CarList_update_info_core(CarList_list[list_th], car_env.UPDATE_TIME_PER_DIDA)
                turns += car_env.UPDATE_TIME_PER_DIDA

            # 判断仿真是否结束
            if time_tag >= car_env.TIME_TAG_UP_BOUND:
                done = True
            if done:
                finish_flag = True

        ###### 开始输出计算的结果 ######
        global output_title_flag
        global output_file
        farest_car_location_y = 0.0
        if output_title_flag == False:
            output_title_flag = True
            output_file.write('*********************************************************************************'+'\n')
            output_file.write('Output Format: time_tag,ID,x,y,v_x,v_y,max_v,acc_x,acc_y,cur_lane,from_lane,to_lane' + '\n')
            output_file.write('TIME_TAG_UP_BOUND%.2f' % car_env.TIME_TAG_UP_BOUND + '\n')
            output_file.write('STRATEGY = ' + STRATEGY + '\n')
            output_file.write(
                'left_max:%.2f' % car_env.left_max_v + ',mid_max:%.2f' % car_env.mid_max_v + ',right_max:%.2f' % car_env.right_max_v + '\n')
            output_file.write('Car lenght: 5m, lane width: 3.5m\n')
            output_file.write('*********************************************************************************'+'\n')
        for th_ in range(len(vehicle_list)):
            if int(time_tag/car_env.AI_DT%output_interval) == 0:
                oneline = '%.2f' % time_tag + ',%d' % vehicle_list[th_].id + ',%.2f' % vehicle_list[
                    th_].location_x + ',%.2f' % vehicle_list[th_].location[1] + ',%.2f' % vehicle_list[
                              th_].speed_x + ',%.2f' % vehicle_list[th_].speed + ',%.2f' % vehicle_list[
                              th_].max_v + ',%.2f' % vehicle_list[th_].acc_x + ',%.2f' % vehicle_list[th_].acc + ',%d' % \
                          vehicle_list[th_].cur_lane + ',%d' % vehicle_list[th_].from_lane + ',%d' % vehicle_list[
                              th_].to_lane
                output_file.write(oneline + '\n')
            # 找到跑的最远的车
            if vehicle_list[th_].location[1] > farest_car_location_y:
                farest_car_location_y = vehicle_list[th_].location[1]
        # 输出platoon内部的间距变化
        global output_platoon_title_flag
        global output_platoon_spacing_file
        if output_platoon_title_flag == False:
            output_platoon_title_flag = True
            output_platoon_spacing_file.write('*********************************************************************************' + '\n')
            output_platoon_spacing_file.write('TIME_TAG_UP_BOUND%.2f' % car_env.TIME_TAG_UP_BOUND + '\n')
            output_platoon_spacing_file.write('STRATEGY = ' + STRATEGY + '\n')
            output_platoon_spacing_file.write('Format: time_tag,platoon_id,mean_interval' + '\n')
            output_platoon_spacing_file.write('Car lenght: 5m, lane width: 3.5m\n')
            output_platoon_spacing_file.write('*********************************************************************************' + '\n')
        for th_ in range(len(CarList_list)):
            temp_carlist = CarList_list[th_]
            temp_mean_spacing = 0
            if len(temp_carlist) > 2:
                for i in range(len(temp_carlist)-1):
                    if i == 0:
                        continue
                    temp_mean_spacing += (temp_carlist[i].location[1]-temp_carlist[i+1].location[1])
                temp_mean_spacing = temp_mean_spacing/(len(temp_carlist)-1)
                # 输出队列内部平均间隔
                if int(time_tag/car_env.AI_DT%output_interval) == 0:
                    oneline = '%.2f' % time_tag + ',%d' % th_ + ',%.2f'%temp_mean_spacing
                    output_platoon_spacing_file.write(oneline + '\n')

        if int(time_tag/car_env.AI_DT%output_interval) == 0:
            print('Now time: %.2f' %time_tag + ', farest car location_y: %.2f' % farest_car_location_y)
        ###### 有强化学习的车跑到了终点，结束仿真 ######
        if finish_flag:
            # output_file.write('============== Finish info:' + finish_info + '============== ')
            output_platoon_spacing_file.close()
            output_file.close()
            break





if __name__ == '__main__':
    if STRATEGY_INPUT == 'RL':
        # limit graphic RAM usage
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.28  # 至少一块卡上留70%的显存，保证5个进程能跑起来
        sess = tf.Session(config=config)
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
        # path = './' + 'NN_Data/3_cars_following/'
        path = 'NN/3_car3_following_good_networks/COTA2018-1st-wave/graph-20171026-1536'
        saver.restore(sess, tf.train.latest_checkpoint(path))

    #### 开始执行交通流测试
    eval_trafficFlow(STRATEGY=STRATEGY_INPUT)


