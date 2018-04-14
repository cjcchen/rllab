from sandbox.rocky.tf.algos.ddpg.ddpg import DDPG

import gym
import numpy as np
import tensorflow as tf

RANDOM_SEED = 1234

np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
env.seed(RANDOM_SEED)

env = gym.make('Pendulum-v0')
ddpg = DDPG(env, plot=True, log_dir="cartpole")

ddpg.train()
