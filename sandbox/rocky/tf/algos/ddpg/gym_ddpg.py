from ddpg import DDPG

import gym
import numpy as np
import tensorflow as tf

RANDOM_SEED = 1234
env = gym.make('CartPole-v0')

print (env.action_space)
#env = gym.make('Pendulum-v0')

np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
env.seed(RANDOM_SEED)

ddpg = DDPG(env, log_dir="cartpole")

ddpg.train()
