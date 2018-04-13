import gym
from ddpg import DDPG
#from noise import OrnsteinUhlenbeckProcess
import numpy as np
import ddpg_config
import tensorflow as tf



RANDOM_SEED = 1234

env = gym.make('Pendulum-v0')

np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
env.seed(RANDOM_SEED)

ddpg=DDPG(env)

ddpg.train()

