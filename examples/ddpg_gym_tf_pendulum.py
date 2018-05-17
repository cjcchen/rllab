from sandbox.rocky.tf.algos.ddpg.ddpg import DDPG
from sandbox.rocky.tf.algos.ddpg.noise import OrnsteinUhlenbeckActionNoise
from rllab.misc import ext

import gym
import numpy as np
import tensorflow as tf

RANDOM_SEED = 1234

ext.set_seed(RANDOM_SEED)

env = gym.make('Pendulum-v0')
env.seed(ext.get_seed())


action_dim = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(
    mu=np.zeros(action_dim), sigma=float(0.02) * np.ones(action_dim))

ddpg = DDPG(
    env,
    plot=False,
    action_noise=action_noise,
    check_point_dir='pendulum',
    log_dir="pendulum_ou_noise")

ddpg.train()
