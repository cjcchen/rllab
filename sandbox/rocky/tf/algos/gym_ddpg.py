import gym
from ddpg import DDPG
from noise import OrnsteinUhlenbeckProcess
import numpy as np
import ddpg_config
import tensorflow as tf



RANDOM_SEED = 1234

env = gym.make('Pendulum-v0')

np.random.seed(RANDOM_SEED)
tf.set_random_seed(RANDOM_SEED)
env.seed(RANDOM_SEED)

ddpg=DDPG(
        env,
        policy = None,
        qf = None,
        batch_size=64,
        n_epochs=1000000,
        epoch_length=10000,
        min_pool_size=10000,
        replay_pool_size=1000000,
        discount=0.99,
        soft_target_tau=0.001,
        scale_reward=1.0,
        plot=False,
        log_dir="ddpg_test2")

ddpg.train()

