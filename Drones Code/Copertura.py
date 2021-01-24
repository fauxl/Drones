import matplotlib.pyplot as plt
import random
import math
import numpy as np
import matplotlib as mpl
import matplotlib.animation as animation

import gym
import tensorflow as tf
from PIL import Image, ImageDraw
from UtilityAgent import *
from DroneEnv import *
from Graphics import *

try:
    from pyglet.gl import gl_info
    openai_cart_pole_rendering = True   # no problem, let's use OpenAI gym's rendering function
except Exception:
    openai_cart_pole_rendering = False  # probably no X server available, let's use our own rendering function


env = DroneEnv(30)
obs = env.reset()

"""
def render_cart_pole(env, obs):
    if openai_cart_pole_rendering:
        # use OpenAI gym's rendering function
        return env.render()

def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,

def plot_animation(frames, repeat=False, interval=60):
    plt.close()  # or else nbagg sometimes plots in the previous cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames), repeat=repeat, interval=interval)

"""
tf.compat.v1.disable_eager_execution()

"""

## 1. Specify the network architecture
n_inputs = 5  # == env.observation_space.shape[0]
n_hidden = 10  # it's a simple task, we don't need more than this
n_outputs = 8 # only outputs the probability of accelerating left

learning_rate = 0.01

initializer = tf.keras.initializers.VarianceScaling()

# 2. Build the neural network
X = tf.compat.v1.placeholder(tf.float32, shape=[None, n_inputs])

hidden = tf.keras.layers.Dense(n_hidden, activation=tf.nn.elu, kernel_initializer=initializer) (X)
logits = tf.keras.layers.Dense(n_outputs)(hidden)
outputs = tf.nn.sigmoid(logits)

# 3. Select a random action based on the estimated probabilities
prob = tf.concat(axis=8, values=outputs)
action = tf.random.categorical(tf.math.log(prob), num_samples=8)

y = 8. - tf.cast(action,float)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cross_entropy)

grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.compat.v1.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.compat.v1.global_variables_initializer()
saver = tf.compat.v1.train.Saver()


def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]
   
print(discount_rewards([10, 0, -50], discount_rate=0.8))

print(discount_and_normalize_rewards([[10, 0, -50], [10, 20]], discount_rate=0.8))

env = DroneEnv(30)

n_games_per_update = 10
n_max_steps = 5000
n_iterations = 200
save_iterations = 10
discount_rate = 0.95

with tf.compat.v1.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        print("\rIteration: {}".format(iteration), end="")
        all_rewards = []
        all_gradients = []
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()
            for step in range(n_max_steps):
                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
        feed_dict = {}
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
                                      for game_index, rewards in enumerate(all_rewards)
                                          for step, reward in enumerate(rewards)], axis=0)
            feed_dict[gradient_placeholder] = mean_gradients
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, "./my_policy_net_pg.ckpt")

env.close()
"""
"""
n_environments = 5
n_iterations = 100

envs = [gym.make("CartPole-v0") for _ in range(n_environments)]
observations = [env.reset() for env in envs]

with tf.compat.v1.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        target_probas = np.array([([1.] if obs[2] < 0 else [0.]) for obs in observations]) # if angle<0 we want proba(left)=1., or else proba(left)=0.
        action_val, _ = sess.run([action, training_op], feed_dict={X: np.array(observations), y: target_probas})
        for env_index, env in enumerate(envs):
            obs, reward, done, info = env.step(action_val[env_index][0])
            observations[env_index] = obs if not done else env.reset()
    saver.save(sess, "./my_policy_net_basic.ckpt")

for env in envs:
    env.close()


def render_policy_net(model_path, action, X, n_max_steps = 1000):
    frames = []
    env = DroneEnv(20)
    obs = env.reset()
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, model_path)
        for step in range(n_max_steps):
            #env.render()
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                break
    env.close()
    return frames


frames = render_policy_net("./my_policy_net_pg.ckpt", action, X)
print(action)
env.render()
plt.show()
env.close()
"""