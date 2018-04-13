from copy import copy

from ddpg_base_net import ActorNet, CriticNet
from replay_buffer import ReplayBuffer
from noise import OrnsteinUhlenbeckActionNoise
#from noise1 import OrnsteinUhlenbeckProcess

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc



class DDPG(object):
    def __init__(self,env, gamma = 0.99, tau = 0.01, observation_range= (-5,5), action_range= (-1,1), 
        actor_lr = 1e-4, critic_lr = 1e-3, reward_scale = 1, batch_size = 64, critic_l2_weight_decay = 0.01,
        clip_norm = None, log_dir="test2"
        ):

        self._env = env
        self._session = tf.Session()

        observation_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        nb_actions =  nb_actions = env.action_space.shape[-1]
        
        print ("nb actions:",nb_actions)
        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(0.02) * np.ones(nb_actions))

        # Inputs.
        self._state = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='state')
        self._next_state = tf.placeholder(tf.float32, shape=(None,) + observation_shape, name='next_state')
        self._terminals = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
        self._rewards = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
        self._actions = tf.placeholder(tf.float32, shape=(None,) + action_shape, name='actions')
        self._critic_target = tf.placeholder(tf.float32, shape=(None, 1), name='critic_target')

        self._observation_shape = observation_shape
        self._action_shape = action_shape

        # Parameters.
        self._tau = tau
        self._action_noise = action_noise
        self._action_range = action_range
        self._clip_norm = clip_norm
        self._reward_scale = reward_scale
        self._batch_size = batch_size


        state_dim = observation_shape[0]
        action_dim = observation_shape[-1]

        #actor
        self._actor_net = ActorNet(self._session, nb_actions, lr = actor_lr)
        self._target_actor = copy(self._actor_net)
        self._target_actor.name = 'target_actor'
        

        #critic
        self._critic_net = CriticNet(self._session, gamma = gamma, lr = critic_lr, weight_decay = critic_l2_weight_decay)
        self._target_critic = copy(self._critic_net)
        self._target_critic.name = 'target_critic'

        #replay buffer
        self._replay_buffer = ReplayBuffer(1e6)

        self._summary_writer = tf.summary.FileWriter(log_dir, self._session.graph)
        self._initialize()

    def _initialize(self):
        #build network
        self._actor_net.build_net(self._state)
        self._target_actor.build_net(self._next_state)

        self._critic_net.build_net(self._state, self._actions, self._rewards, self._terminals, self._critic_target, self._actor_net.action)
        self._target_critic.build_net(self._next_state, self._actions, self._rewards, self._terminals, self._critic_target, self._target_actor.action)

        #calculate critic grade
        grads = tf.gradients(self._critic_net.loss, self._critic_net.trainable_vars)
        if(self._clip_norm):
            grads, _ = tf.clip_by_global_norm(grads, self._clip_norm) # gradient clipping
        grads_and_vars = list(zip(grads, self._critic_net.trainable_vars))
        self._critic_train_op = self._critic_net.optimizer.apply_gradients(grads_and_vars)

        #calculate actor grade
        a_grads = tf.gradients(self._critic_net.action_loss, self._actor_net.trainable_vars)
        if(self._clip_norm):
            a_grads, _ = tf.clip_by_global_norm(a_grads, self._clip_norm) # gradient clipping
        a_grads_and_vars = list(zip(a_grads, self._actor_net.trainable_vars))
        self._actor_train_op = self._actor_net.optimizer.apply_gradients(a_grads_and_vars)

        #setup network to target network params update op 
        self._actor_net.setup_target_net(self._target_actor, self._tau)
        self._critic_net.setup_target_net(self._target_critic, self._tau)

        self._session.run(tf.global_variables_initializer())

    def _train_net(self):
        # Get a batch.
        [state, action, reward, terminal, next_state] = self._replay_buffer.get_batch_data(self._batch_size)
        reward = reward.reshape(-1,1)
        terminal = terminal.reshape(-1,1)

        target_Q = self._target_critic.predict_target_Q(next_state, reward, terminal)

        ops = [self._actor_train_op, self._critic_train_op]
        self._session.run(ops, feed_dict={
            self._state: state,
            self._actions: action,
            self._critic_target: target_Q,
        })
    
        self._actor_net.update_target_net()
        self._critic_net.update_target_net()
        return 

    def _report_total_reward(self, reward, step):
        summary = tf.Summary()
        summary.value.add(tag='rollout/reward', simple_value=float(reward))
        summary.value.add(tag='train/episode_reward', simple_value=float(reward))
        self._summary_writer.add_summary(summary, step) 

    def predict(self, state):
        action = self._actor_net.predict(np.array(state).reshape(1,-1))[0]
        noise = self._action_noise.gen()
        action = action + noise
        action = np.clip(action, self._action_range[0], self._action_range[1])
        return action

    def train(self, epochs=500, epoch_cycles=20, rollout_steps=100, train_steps=50):
        state=self._env.reset()
        self._action_noise.reset()
        total_reward = 0.0
        reward_list = []
        cyc_round = 0
        max_action = self._env.action_space.high
        for epoch in range(epochs):
            for step in range(epoch_cycles):
                for rollout in range(rollout_steps):
                    #self._env.render()
                    action = self.predict(state) 

                    next_state, reward, terminal, info = self._env.step(action*max_action)
                    self._replay_buffer.add_data(state,action,reward*self._reward_scale,terminal, next_state)
                    state = next_state
                    total_reward += reward
                    if terminal:
                        self._report_total_reward(total_reward, cyc_round)
                        print ("epoch %d, total reward %lf\n" %(cyc_round, total_reward))
                        cyc_round +=1
                        total_reward = 0
                        state=self._env.reset()
                        self._action_noise.reset()

                for train in range(train_steps):
                    self._train_net()


