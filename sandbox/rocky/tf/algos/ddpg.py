
from ddpg_base_net import ActorNet, CriticNet
from noise import OrnsteinUhlenbeckProcess
from replay_buffer import ReplayBuffer

import ddpg_config 
import tensorflow as tf
import numpy as np
import random

OU_THETA = 0.15
OU_MU = 0.
OU_SIGMA = 0.3

class DDPG:
    def __init__(
            self,
            env,
            policy = None,
            qf = None,
            batch_size=32,
            n_epochs=1000000,
            epoch_length=10000,
            min_pool_size=10000,
            replay_pool_size=1000000,
            discount=0.99,
            soft_target=True,
            soft_target_tau=0.001,
            scale_reward=1.0,
            noise_stddev=0.0001,
            plot=False,
            log_dir=None):

        self._env = env
        self._batch_size=batch_size
        self._n_epochs=n_epochs
        self._epoch_length=epoch_length
        self._min_pool_size=min_pool_size
        self._replay_pool_size=replay_pool_size
        self._discount=discount
        self._soft_target=soft_target
        self._soft_target_tau=soft_target_tau
        self._scale_reward=scale_reward

        self._env =  env

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        self._session = tf.Session()

        with tf.variable_scope('actor_net'):
            self._actor_net = ActorNet(self._session, state_dim, action_dim, action_bound )
        with tf.variable_scope('critic_net'):
            self._critic_net = CriticNet(self._session, state_dim, action_dim)

        nb_actions = env.action_space.shape[-1]
        #self._noise = OrnsteinUhlenbeckProcess(mu=np.zeros(nb_actions), sigma=float(noise_stddev)*np.ones(nb_actions))
        self._noise = OrnsteinUhlenbeckProcess(ddpg_config.OU_THETA, mu=ddpg_config.OU_MU, sigma=ddpg_config.OU_SIGMA, n_steps_annealing=ddpg_config.EXPLORATION_EPISODES)
        self._replay_buffer = ReplayBuffer(self._replay_pool_size)

        self._global_step = 0
        self._batch_size = 64

        self._summary_writer = tf.summary.FileWriter(log_dir, self._session.graph)
        self._session.run(tf.global_variables_initializer())

    def train(self):
        for epoch in range(self._n_epochs):
            state=self._env.reset()
            #self._noise.reset()
            total_reward = 0.0

            for step in range(self._epoch_length):
                self._env.render()
                if self._global_step < self._min_pool_size:
                    action = self._env.action_space.sample()
                else:
                    #action = np.clip(self._actor_net.predict(np.array(state).reshape(1,-1))[0] + self._noise.generate(), -1, 1) 
                    action = np.clip(self._actor_net.predict(np.array(state).reshape(1,-1))[0] + self._noise.generate(step), -1, 1) 
            
                next_state, reward, terminal, info = self._env.step(action) 
                self._replay_buffer.add_data(state,action,reward*self._scale_reward,terminal, next_state)
                self.train_net()

                state = next_state
                total_reward += reward
                self._global_step +=1
                if terminal:
                    break

            self.report_total_reward(total_reward, epoch)
            print ("epoch %d, total reward %lf\n" %(epoch, total_reward))

    def get_action(self, state):
        return self._actor_net.predict(np.array(state).reshape(1,-1))

    def train_net(self):
        if self._replay_buffer.get_buffer_size() > self._min_pool_size:
            [state, action, reward, terminal, next_state] = self._replay_buffer.get_batch_data(self._batch_size)

            target_q_value = self.calculate_target(reward, terminal, next_state)
            critic_summary, loss,_=self._critic_net.train( state, action, target_q_value )

            action_out = self._actor_net.predict(state)
            action_grads = self._critic_net.action_gradients(state, action_out)
        
            actor_summary,_=self._actor_net.train(state, action_grads[0])

            self._actor_net.update_target_net()
            self._critic_net.update_target_net()

            summary = tf.Summary()
            summary.value.add(tag='rollout/loss', simple_value=loss)
            self._summary_writer.add_summary(summary, self._global_step) 
        
            self._summary_writer.add_summary(actor_summary, self._global_step) 
            self._summary_writer.add_summary(critic_summary, self._global_step) 
            
            if self._global_step % 100==0:
                print ("train loss: %f, buffer size %d\n"%(loss,self._replay_buffer.get_buffer_size())) 
                #print ("train loss: %f, buffer size %d\n"%(loss,len(self.replay_buffer))) 
                 

    def calculate_target(self, rewards, terminals, next_states):
        next_state_action = self._actor_net.predict_target(next_states)
        next_state_q_value = self._critic_net.predict_target(next_states, next_state_action)

        new_target = rewards + self._discount * next_state_q_value * (1-terminals.astype(float))
        targets = []
        for i, r in enumerate(rewards):
            if terminals[i]:
                targets.append(r)
            else:
                targets.append(r + self._discount * next_state_q_value[i])
        targets = np.array(targets).reshape(-1,1)
        assert targets.shape[1] == 1
        return targets

    def report_total_reward(self,val, step):
        summary = tf.Summary()
        summary.value.add(tag='rollout/reward', simple_value=float(val))
        summary.value.add(tag='train/episode_reward', simple_value=float(val))
        self._summary_writer.add_summary(summary, step) 
        print ("step :%d reward %d, buffer size %d\n" %(step, val, self._replay_buffer.get_buffer_size()))

