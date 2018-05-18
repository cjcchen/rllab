from sandbox.rocky.tf.algos.ddpg.ddpg import DDPG
from sandbox.rocky.tf.algos.ddpg.noise import OrnsteinUhlenbeckActionNoise
from sandbox.rocky.tf.algos.network.actor_critic_net import ActorNet, CriticNet
from rllab.misc.instrument import run_experiment_lite
from rllab.misc import ext

import gym
import numpy as np
import tensorflow as tf

RANDOM_SEED = 1234

ext.set_seed(RANDOM_SEED)


def run_task(*_):

    env = gym.make('Pendulum-v0')
    env.seed(ext.get_seed())

    action_dim = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(action_dim), sigma=float(0.02) * np.ones(action_dim))

    actor_net = ActorNet(
        action_dim=env.action_space.shape[-1], lr=1e-4, hidden_layers=[64, 64])

    critic_net = CriticNet(
        gamma=0.99, lr=1e-3, weight_decay=0.01, hidden_layers=[64, 64])

    ddpg = DDPG(
        env,
        actor_net=actor_net,
        critic_net=critic_net,
        plot=False,
        action_noise=action_noise,
        check_point_dir='pendulum',
        log_dir="pendulum_ou_noise")

    ddpg.train()


run_experiment_lite(
    run_task,
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    seed=1,
    # plot=True,
)
