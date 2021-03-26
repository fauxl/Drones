import random
import math
import os
import numpy as np
import time

import gym

from collections import deque
import tensorflow as tf
from UtilityAgent import *
from DroneEnv import *
from DroneSignalEnv import *
from Graphics import *

tf.compat.v1.disable_eager_execution()

size = 50
env = DroneEnv(size)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

class QNetwork():
    def __init__(self, state_dim, action_size):
        self.state_in = tf.compat.v1.placeholder(tf.float32, shape=[None, *state_dim])
        self.action_in = tf.compat.v1.placeholder(tf.int32, shape=[None])
        self.q_target_in = tf.compat.v1.placeholder(tf.float32, shape=[None])
        action_one_hot = tf.one_hot(self.action_in, depth=action_size)
        
        #network
        self.hidden1 = tf.compat.v1.layers.dense(self.state_in,128, activation=tf.nn.relu)
        self.q_state = tf.compat.v1.layers.dense(self.hidden1, action_size, activation=None)
        self.q_state_action = tf.compat.v1.reduce_sum(tf.multiply(self.q_state, action_one_hot), axis=1)
        
        #loss and optimizer function
        self.loss = tf.compat.v1.reduce_mean(tf.square(self.q_state_action - self.q_target_in))
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        
    def update_model(self, session, state, action, q_target):
        feed = {self.state_in: state, self.action_in: action, self.q_target_in: q_target}
        session.run(self.optimizer, feed_dict=feed)
           
    def get_q_state(self, session, state):
        q_state = session.run(self.q_state, feed_dict={self.state_in: state})
        return q_state

# replay same action performed in the past
class ReplayBuffer():
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        sample_size = min(len(self.buffer), batch_size)
        #print(self.buffer)

        #for item in self.buffer
        samples = random.choices(self.buffer, k=sample_size)
        return map(list, zip(*samples))

class DQNAgent():
    def __init__(self, env):
        self.state_dim = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = QNetwork(self.state_dim, self.action_size)
        self.replay_buffer = ReplayBuffer(maxlen=200)
        self.gamma = 0.98       #discount
        self.eps = 1.0          #randomfactor
        
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        
    def get_action(self, state):
        q_state = self.q_network.get_q_state(self.sess, [state])
        action_greedy = np.argmax(q_state)
        action_random = np.random.randint(self.action_size)
        action = action_random if random.random() < self.eps else action_greedy
        newi,newj = env.try_step(action)

        # avoids the drone from getting out of the map, removing some performable action
        if(newi>size-1 or newj>size-1):
            if(action != 1 or action != 2 or action !=5):
                action =  random.choice([1, 2,5])
    
        if(newi<0):
            action =  3
        
        if(newj<0):
            action =  0

        if(newi>size-1 and newj<=2):
            action =  4
        
        if(newj>size-1 and newi<=2):
            action =  7

        return action

    # alg training
    def train(self, state, action, next_state, reward, done,info):
        
        self.replay_buffer.add((state, action, next_state, reward, done))
        states, actions, next_states, rewards, dones = self.replay_buffer.sample(200)
        q_next_states = self.q_network.get_q_state(self.sess, next_states)
        q_next_states[dones] = np.zeros([self.action_size])

        # adjust discount to stick to the path once found
        if info == 1 and self.gamma == 0.98:
            self.gamma = 1.2
        if self.gamma > 0.98:
            self.gamma = self.gamma*0.9999988

        q_targets = rewards + self.gamma* np.max(q_next_states, axis=1)
        self.q_network.update_model(self.sess, states, actions, q_targets)
        
        # lower casual action while learning
        if done:
             self.eps = max(0.1,0.995*self.eps)
    
    def __del__(self):
        self.sess.close()


agent = DQNAgent(env)
num_episodes = 500
totalRewards =np.zeros(num_episodes)
total_reward = 0
succeed = 0
savenv = env
index = 0
eps_history = []

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

        # count how many times the drone got to the destination
        if info == 1:
            succeed +=1

    eps_history.append(agent.eps)
    print("Episode: {}, total_reward: {:.2f}, eps: {}".format(ep, total_reward, agent.eps))

# plot DroneEnv and statistics
savenv.render()
plt.show()
print("Arrivato:", succeed)
filename = 'img/Rewards&Eps&Episodes.png'
x = [i+1 for i in range(num_episodes)]
plotLearning(x, totalRewards, eps_history, filename)

