# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Flood Sung
# Date: 2016.5.4
# -----------------------------------
import gym
import tensorflow as tf
import numpy as np
from .ou_noise import OUNoise
from .critic_network import CriticNetwork
from .actor_network_bn import ActorNetwork
from .replay_buffer import ReplayBuffer

# training Hyper Parameters:
REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.99
# network hyper-parameters
LAYER1_SIZE = 400
LAYER2_SIZE = 300
actor_LEARNING_RATE = 1e-4
critic_LEARNING_RATE = 1e-3
TAU = 0.001
critic_L2_REG = 0.01


class DDPG:
    """docstring for DDPG"""
    def __init__(self, state_dim, action_dim, action_bound):
        self.name = 'DDPG' # name for uploading results
        # self.environment = env

        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        # limit graphic ram
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 至少一块卡上留70%的显存，保证5个进程能跑起来
        self.sess = tf.InteractiveSession(config=config)

        # build networks
        self.actor_network = ActorNetwork(self.sess, self.state_dim, self.action_dim, self.action_bound, LAYER1_SIZE,
                                          LAYER2_SIZE, TAU, actor_LEARNING_RATE)
        self.critic_network = CriticNetwork(self.sess, self.state_dim, self.action_dim, LAYER1_SIZE, LAYER2_SIZE, TAU,
                                            critic_LEARNING_RATE, critic_L2_REG)
        
        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim, mu=0, theta=0.15, sigma=0.2)

    def train(self):
        #print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

        # Calculate y_batch
        
        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        y_batch = []  
        for i in range(len(minibatch)): 
            if done_batch[i]:
                y_batch.append(reward_batch[i])
            else :
                y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])
        y_batch = np.resize(y_batch,[BATCH_SIZE,1])
        # Update critic by minimizing the loss L
        self.critic_network.train(y_batch,state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)

        self.actor_network.train(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def noise_action(self, state):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        return action + self.exploration_noise.noise()

    def action(self, state):
        action = self.actor_network.action(state)
        return action

    def perceive(self, state, action, reward, next_state, done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.train()

        # if self.time_step % 10000 == 0:
        # self.actor_network.save_network(self.time_step)
        # self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()
