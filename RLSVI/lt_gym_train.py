import gym
import pickle
# import Image
import numpy as np
from RLSVI.rlsvi import *
# import tensorflow as tf
import cv2

# Extract Y channel from RGB, resize to 84x84 and return
# def toNatureDQNFormat(frame):
#     return np.array(Image.fromarray(frame).convert('YCbCr').resize((84, 84), Image.BILINEAR))[:, :, 0] / np.float32(255.0)

#
def RLVSI_preprocess(frame):
    # colframe = np.array(np.resize(frame, (42, 42)))
    colframe = cv2.resize(frame,  (42, 42), interpolation=cv2.INTER_AREA) / 255.0
    return np.concatenate([colframe.flatten(), [1]])


import gym

# env = gym.make('IceHockey-v0')
env = gym.make('MountainCar-v0')

num_episodes = 500
num_steps = 15000

na = env.action_space.n
ns = 42 * 42 * 3 + 1

print(f'number of episodes: {na}, number of steps: {ns}')

sigma = 1.0
lmb = 1.0
gamma = 0.99

R = rlsvi(ns, na, sigma, lmb)

# sess = tf.InteractiveSession()
# try:
#     sess.run(tf.global_variables_initializer())
#
# except AttributeError:
#     sess.run(tf.initialize_all_variables())

for episode in range(num_episodes):
    obs = env.reset()
    obs_pp = RLVSI_preprocess(obs)
    # obs_pp = obs
    for time_step in range(num_steps):
        if time_step % 1000 == 0:
            print(f'Episode {episode}, Time step {time_step}')
        # env.render()
        action = R.choose_action(obs_pp)
        next_obs, reward, done, info = env.step(action)
        next_obs_pp = RLVSI_preprocess(next_obs)
        # next_obs_pp = next_obs
        R.add_data(obs_pp, action, reward, next_obs_pp, done)
        obs_pp = next_obs_pp
        if reward != 0:
            print(reward)
        if done:
            break
    R.new_episode()
