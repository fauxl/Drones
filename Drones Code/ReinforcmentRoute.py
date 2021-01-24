import matplotlib.pyplot as plt
import random
import math
import os
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation
import time

from tensorflow.python.keras.models import Model
import gym

from collections import deque
from IPython.display import clear_output
from gym.envs.registration import register
import tensorflow as tf
from PIL import Image, ImageDraw
from UtilityAgent import *
from DroneEnv import *
from DroneSignalEnv import *
from Graphics import *

tf.compat.v1.disable_eager_execution()


env = DroneSignalEnv(50)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

""""
class Agent():
    def __init__(self, env):
        self.is_discrete = 
            type(env.action_space) == gym.spaces.discrete.Discrete

        self.action_size = env.action_space.n
        print("Action size:", self.action_size)
        
    def get_action(self, state):
        action = random.choice(range(self.action_size))
        return action


class QNAgent(Agent):
    def __init__(self, env, discount_rate=0.95, learning_rate=0.001):
        super().__init__(env)
        self.state_size = env.size*env.size
        print("State size:", self.state_size)
        
        self.action_size = env.action_space.n
        self.state_dim = env.observation_space.shape
        self.eps = 1.0
        self.i = 0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model(self.state_dim,self.action_size)
        
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    def build_model(self,state_dim,action_size):
        tf.compat.v1.reset_default_graph()

        self.state_in = tf.compat.v1.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.compat.v1.placeholder(tf.float32, shape=[None])
        
        self.hidden1 = tf.compat.v1.layers.dense(self.state_in, 100, activation=tf.nn.relu)
        self.action = tf.compat.v1.one_hot(self.action_in, depth=self.action_size)
        
        self.q_state = tf.compat.v1.layers.dense(self.hidden1, action_size, name="q_table")
        self.q_action = tf.compat.v1.reduce_sum(tf.multiply(self.q_state, self.action), axis=1)
        
        self.loss = tf.compat.v1.reduce_sum(tf.square(self.q_target_in - self.q_action))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
    def get_action(self, state):
        q_state = self.sess.run(self.q_state, feed_dict={self.state_in: [state]})
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.eps else action_greedy
    
    def train(self, experience):
        state, action, next_state, reward, done, info = ([exp] for exp in experience)
        
        q_next = self.sess.run(self.q_state, feed_dict={self.state_in: next_state})
        q_next[done] = np.zeros([self.action_size])
        q_target = reward + self.discount_rate * np.max(q_next)
        
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
        self.sess.run(self.optimizer, feed_dict=feed)
        
        if experience[4]:
            self.eps = self.eps * 0.99
            
    def __del__(self):
        self.sess.close()
        
agent = QNAgent(env)

num_episodes = 500
total_reward = 0
totalRewards = np.zeros(num_episodes)
for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train((state,action,next_state,reward,done,info))
        state = next_state
        total_reward += reward
        totalRewards[ep] = total_reward    

        #print("s:", state, "a:", action)
        #print("Episode: {}, reward: {}, Total reward: {}, eps: {}".format(ep,reward, total_reward,agent.eps), end="")
        
        with tf.compat.v1.variable_scope("q_table", reuse=True):
            weights = agent.sess.run(tf.compat.v1.get_variable("kernel"))
            #print(weights)
    print("\rEpisode: {}, reward: {}, Total reward: {}, eps: {},info: {}".format(ep,reward, total_reward,agent.eps,info), end="")
env.render()
plt.plot(totalRewards)
plt.show()

"""

class QNetwork():
    def __init__(self, state_dim, action_size):
        self.state_in = tf.compat.v1.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.compat.v1.placeholder(tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)
        
        self.hidden1 = tf.compat.v1.layers.dense(self.state_in, 128, activation=tf.nn.relu)
        self.q_state = tf.compat.v1.layers.dense(self.hidden1, action_size, activation=None)
        self.q_state_action = tf.compat.v1.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)
        
        self.loss = tf.compat.v1.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        
    def update_model(self, session, state, action, q_target):
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
        session.run(self.optimizer, feed_dict=feed)
           
    def get_q_state(self, session, state):
        q_state = session.run(self.q_state, feed_dict={self.state_in: state})
        return q_state

class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        samples = random.choices(self.buffer, k=sample_size)
        return map(list, zip(*samples))

class DQNAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = QNetwork(self.state_dim, self.action_size)
        self.replay_buffer = ReplayBuffer(maxlen=10000)
        self.gamma = 0.98  #discount
        self.eps = 1.0
        
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        action_greedy = np.argmax(q_state)
        action_random = np.random.randint(self.action_size)
        action = action_random if random.random() < self.eps else action_greedy
        return action

    def train(self, state, action, next_state, reward, done,info):

        self.replay_buffer.add((state, action, next_state, reward, done))
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(200)
        q_next_states = self.q_network.get_q_state(self.sess, next_states)
        q_next_states[dones] = np.zeros([self.action_size])
        q_targets = rewards + self.gamma * np.max(q_next_states, axis=1)
        self.q_network.update_model(self.sess, states, actions, q_targets)
        
        if done: self.eps = max(0.1,0.99*self.eps)
    
    def __del__(self):
        self.sess.close()

#env = DroneSignalEnv(50)
agent = DQNAgent(env)
num_episodes = 600
totalRewards = np.zeros(num_episodes)
total_reward = 0

for ep in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.train(state, action, next_state, reward, done, info)
        total_reward += reward
        state = next_state
        totalRewards[ep] = total_reward
        
    print("Episode: {}, total_reward: {:.2f}, eps: {}".format(ep, total_reward, agent.eps))

env.render()
plt.plot(totalRewards)
plt.show()


"""
 ////

class Agent():
    def __init__(self, env):
        self.is_discrete = \
            type(env.action_space) == gym.spaces.discrete.Discrete
        
        if self.is_discrete:
            self.action_size = env.action_space.n
            print("Action size:", self.action_size)
        else:
            self.action_low = env.action_space.low
            self.action_high = env.action_space.high
            self.action_shape = env.action_space.shape
            print("Action range:", self.action_low, self.action_high)
        
    def get_action(self, state):
        if self.is_discrete:
            action = random.choice(range(self.action_size))
        else:
            action = np.random.uniform(self.action_low,
                                       self.action_high,
                                       self.action_shape)
        return action

class QAgent(Agent):
    def __init__(self, env, discount_rate=0.97, learning_rate=0.01):
        super().__init__(env)
        self.state_size = env.state_size
        print("State size:", self.state_size)
        
        self.eps = 1.0
        self.discount_rate = discount_rate
        self.learning_rate = learning_rate
        self.build_model()
        
    def build_model(self):
        self.q_table = 1e-4*np.random.random([self.state_size, self.action_size])

    def get_action(self, state):
        q_state = self.q_table[state]
        print(self.q_table)
        action_greedy = np.argmax(q_state)
        action_random = super().get_action(state)
        return action_random if random.random() < self.eps else action_greedy
    
    def train(self, experience):
        state, action, next_state, reward, done = experience
        
        q_next = self.q_table[next_state]
        q_next = np.zeros([self.action_size]) if done else q_next
        q_target = reward + self.discount_rate * np.max(q_next)
        
        q_update = q_target - self.q_table[state,action]
        self.q_table[state,action] += self.learning_rate * q_update
        
        if done:
            self.eps = self.eps * 0.99
        
agent = QAgent(env)


total_reward = 0
for ep in range(100):
    state = env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        print(next_state)
        agent.train((state,action,next_state,reward,done))
        state = next_state
        total_reward += reward
        
        print("s:", state, "a:", action)
        print("Episode: {}, Total reward: {}, eps: {}".format(ep,total_reward,agent.eps))
        env.render()
        print(agent.q_table)
        #time.sleep(0.05)
        #clear_output(wait=True)

plt.figure(figsize=(11, 7))
plt.subplot(121)
plt.title("Color")
plt.imshow(env)
plt.axis("off")
plt.subplot(122)
plt.title("GreyScale")
plt.imshow(env, interpolation="nearest",cmap='gray')
plt.axis("off")
save_fig("MapConverted")
plt.show()

input_height = 50
input_width = 50
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3 
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n # 9 discrete actions are available
initializer = tf.keras.initializers.VarianceScaling()

def q_network(X_state, name):
    prev_layer = 2500 # scale pixel intensities to the [-1.0, 1.0] range.
    with tf.compat.v1.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.keras.layers.Conv2D(
                filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer,)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

X_state =  tf.compat.v1.placeholder(tf.float32, shape=[None, input_height, input_width,
                                            input_channels])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

online_vars
"""