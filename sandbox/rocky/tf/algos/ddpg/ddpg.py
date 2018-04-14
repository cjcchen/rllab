from sandbox.rocky.tf.algos.ddpg.ddpg_base_net import ActorNet, CriticNet
from sandbox.rocky.tf.algos.ddpg.replay_buffer import ReplayBuffer
from sandbox.rocky.tf.algos.ddpg.noise import OrnsteinUhlenbeckActionNoise

from copy import copy
import numpy as np
import tensorflow as tf


class DDPG(object):
    def __init__(self,
                 env,
                 gamma=0.99,
                 tau=0.01,
                 observation_range=(-5, 5),
                 action_range=(-1, 1),
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 reward_scale=1,
                 batch_size=64,
                 critic_l2_weight_decay=0.01,
                 clip_norm=None,
                 plot=False,
                 log_dir=None):

        self._env = env
        self._session = tf.Session()

        observation_shape = env.observation_space.shape

        action_shape = env.action_space.shape
        actions_dim = env.action_space.shape[-1]
        self._max_action = self._env.action_space.high

        action_noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(actions_dim), sigma=float(0.02) * np.ones(actions_dim))

        # Inputs.
        self._state = tf.placeholder(
            tf.float32, shape=(None, ) + observation_shape, name='state')
        self._next_state = tf.placeholder(
            tf.float32, shape=(None, ) + observation_shape, name='next_state')
        self._terminals = tf.placeholder(
            tf.float32, shape=(None, 1), name='terminals')
        self._rewards = tf.placeholder(
            tf.float32, shape=(None, 1), name='rewards')
        self._actions = tf.placeholder(
            tf.float32, shape=(None, actions_dim), name='actions')
        self._critic_target = tf.placeholder(
            tf.float32, shape=(None, 1), name='critic_target')

        self._observation_shape = observation_shape
        self._action_shape = action_shape

        # Parameters.
        self._tau = tau
        self._action_noise = action_noise
        self._action_range = action_range
        self._clip_norm = clip_norm
        self._reward_scale = reward_scale
        self._batch_size = batch_size
        self._plot = plot

        #actor
        self._actor_net = ActorNet(self._session, actions_dim, lr=actor_lr)
        self._target_actor = copy(self._actor_net)
        self._target_actor.name = 'target_actor'

        #critic
        self._critic_net = CriticNet(
            self._session,
            gamma=gamma,
            lr=critic_lr,
            weight_decay=critic_l2_weight_decay)
        self._target_critic = copy(self._critic_net)
        self._target_critic.name = 'target_critic'

        #replay buffer
        self._replay_buffer = ReplayBuffer(1e6)

        if log_dir:
            self._summary_writer = tf.summary.FileWriter(
                log_dir, self._session.graph)
        else:
            self._summary_writer = None
        self._initialize()

    def _initialize(self):
        #build network
        self._actor_net.build_net(self._state)
        self._target_actor.build_net(self._next_state)

        self._critic_net.build_net(self._state, self._actions, self._rewards,
                                   self._terminals, self._critic_target,
                                   self._actor_net.action)
        self._target_critic.build_net(
            self._next_state, self._actions, self._rewards, self._terminals,
            self._critic_target, self._target_actor.action)

        #set grad chain rule
        self._actor_net.set_grad(self._critic_net.action_grads)

        #setup network to target network params update op
        self._actor_net.setup_target_net(self._target_actor, self._tau)
        self._critic_net.setup_target_net(self._target_critic, self._tau)

        self._session.run(tf.global_variables_initializer())

    def _train_net(self):
        # Get a batch.
        [state, action, reward, terminal,
         next_state] = self._replay_buffer.get_batch_data(self._batch_size)
        reward = reward.reshape(-1, 1)
        terminal = terminal.reshape(-1, 1)

        target_action = self._target_actor.predict(next_state)
        target_Q = self._target_critic.predict_target_Q(
            next_state, target_action, reward, terminal)

        self._critic_net.train(state, action, target_Q)
        self._actor_net.train(state)

        self._actor_net.update_target_net()
        self._critic_net.update_target_net()
        return

    def _report_total_reward(self, reward, step):
        summary = tf.Summary()
        summary.value.add(tag='rollout/reward', simple_value=float(reward))
        summary.value.add(
            tag='train/episode_reward', simple_value=float(reward))
        if self._summary_writer:
            self._summary_writer.add_summary(summary, step)

    def predict(self, state):
        action = self._actor_net.predict(np.array(state).reshape(1, -1))[0]
        noise = self._action_noise.gen()
        action = action + noise
        action = np.clip(action, self._action_range[0], self._action_range[1])
        return action

    def train(self,
              epochs=500,
              epoch_cycles=20,
              rollout_steps=100,
              train_steps=50):
        state = self._env.reset()
        self._action_noise.reset()
        total_reward = 0.0
        cyc_round = 0
        for epoch in range(epochs):
            for step in range(epoch_cycles):
                for rollout in range(rollout_steps):
                    if self._plot:
                        self._env.render()
                    action = self.predict(state)

                    next_state, reward, terminal, info = self._env.step(
                        action * self._max_action)
                    self._replay_buffer.add_data(state, action,
                                                 reward * self._reward_scale,
                                                 terminal, next_state)
                    state = next_state
                    total_reward += reward
                    if terminal:
                        self._report_total_reward(total_reward, cyc_round)
                        print("epoch %d, total reward %lf\n" % (cyc_round,
                                                                total_reward))
                        cyc_round += 1
                        total_reward = 0
                        state = self._env.reset()
                        self._action_noise.reset()

                for train in range(train_steps):
                    self._train_net()
