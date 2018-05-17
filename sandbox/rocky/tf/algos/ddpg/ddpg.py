from sandbox.rocky.tf.algos.network.actor_critic_net import ActorNet, CriticNet
from sandbox.rocky.tf.algos.ddpg.replay_buffer import ReplayBuffer
from rllab.algos.base import RLAlgorithm
from rllab.misc import logger


import tensorflow as tf
from copy import copy
import numpy as np
import os


class DDPG(RLAlgorithm):
    def __init__(self,
                 env,
                 gamma=0.99,
                 tau=0.001,
                 action_range=(-1, 1),
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 reward_scale=1,
                 batch_size=64,
                 critic_l2_weight_decay=0.01,
                 replay_buffer_size=1e6,
                 action_noise=None,
                 plot=False,
                 check_point_dir=None,
                 log_dir=None):
        """
        a DDPG model described in https://arxiv.org/pdf/1509.02971.pdf.
        The hyperparameters used in the model:
        :param env:
        :param gamma: a discount factor 
        :param tau: soft update 
        :param action_range: action space range
        :param actor_lr: learning rate for actor network
        :param critic_lr: learning rate for critic network 
        :param reward_scale: reward discount factor
        :param batch_size: batch size
        :param critic_l2_weight_decay: L2 weight decay for the weights in critic network 
        :param replay_buffer_size: the size of replay buffer capacity
        :param action_noise: custom network for the output mean
        :param plot: Is plot the train process?
        :param checkpoint_dir: directory for saving model 
        :param log_dir: directory for saving tensorboard logs
        :return:
        """

        self._env = env
        self._session = tf.Session()

        observation_shape = env.observation_space.shape

        action_shape = env.action_space.shape
        actions_dim = env.action_space.shape[-1]
        self._max_action = self._env.action_space.high

        # Parameters.
        self._observation_shape = observation_shape
        self._action_shape = action_shape

        self._tau = tau
        self._action_noise = action_noise
        self._action_range = action_range
        self._reward_scale = reward_scale
        self._batch_size = batch_size
        self._plot = plot
        self._check_point_dir = check_point_dir
        self._saver = None

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
        self._replay_buffer = ReplayBuffer(replay_buffer_size)

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

        self._global_step = tf.Variable(
            initial_value=0, name='global_step', trainable=False)

        self.load_session()

    def _train_net(self):
        # Get a batch.
        [state, action, reward, terminal,
         next_state] = self._replay_buffer.get_batch_data(self._batch_size)
        reward = reward.reshape(-1, 1)
        terminal = terminal.reshape(-1, 1)

        target_action = self._target_actor.predict(next_state)
        target_Q = self._target_critic.predict_target_Q(
            next_state, target_action, reward, terminal)

        critic_loss,_=self._critic_net.train(state, action, target_Q)
        self._actor_net.train(state)
        action_loss = self._critic_net.action_loss(state, self._actor_net.predict(state))
        Q_value = self._critic_net.predict_target_Q(
            next_state, action, reward, terminal)

        self._actor_net.update_target_net()
        self._critic_net.update_target_net()
        return action_loss, critic_loss,Q_value

    def predict(self, state):
        action = self._actor_net.predict(np.array(state).reshape(1, -1))[0]
        if self._action_noise:
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
        if self._action_noise:
            self._action_noise.reset()
        total_reward = 0.0
        action_loss_list=[]
        critic_loss_list=[]
        Q_value_list=[]
        episode_step = self._session.run(self._global_step)

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
                        average_action_loss = np.mean(action_loss_list)
                        average_critic_loss = np.mean(critic_loss_list)
                        average_Q_value = np.mean(Q_value_list)
                        max_Q_value = np.max(Q_value_list)


                        action_loss_list=[]
                        critic_loss_list=[]
                        Q_value_list=[]
                     
                        logger.push_prefix('epoch #%d | ' % episode_step)
                        logger.record_tabular('epoch', episode_step)
                        logger.record_tabular('total_reward', total_reward)
                        logger.record_tabular('average action loss', average_action_loss)
                        logger.record_tabular('average critic loss', average_critic_loss)
                        logger.record_tabular('average Q value', average_Q_value)
                        logger.record_tabular('max Q value', max_Q_value)

                        logger.dump_tabular(with_prefix=False)
                        logger.pop_prefix()

                        episode_step = self._session.run(
                            self._global_step.assign_add(1))
                        total_reward = 0
                        state = self._env.reset()
                        if self._action_noise:
                            self._action_noise.reset()

                for train in range(train_steps):
                    action_loss, critic_loss, Q_value = self._train_net()
                    action_loss_list.append(action_loss)
                    critic_loss_list.append(critic_loss)
                    Q_value_list.append(Q_value)
            self.save_session(episode_step)

    def load_session(self):
        if not self._check_point_dir:
            return

        if not self._saver:
            self._saver = tf.train.Saver()
        try:
            print("Trying to restore last checkpoint ...:",
                  self._check_point_dir)
            last_chk_path = tf.train.latest_checkpoint(
                checkpoint_dir=self._check_point_dir)
            self._saver.restore(self._session, save_path=last_chk_path)
            print("restore last checkpoint %s done" % self._check_point_dir)
        except Exception as e:
            if not os.path.exists(self._check_point_dir):
                os.mkdir(self._check_point_dir)
            assert (os.path.exists(
                self._check_point_dir
            )), "%s check point file create fail" % self._check_point_dir
            print(
                "Failed to restore checkpoint. Initializing variables instead."
            ), e
            self._session.run(tf.global_variables_initializer())

    def save_session(self, step):
        if not self._saver:
            return
        save_path = self._check_point_dir + "/event"
        self._saver.save(self._session, save_path=save_path, global_step=step)
