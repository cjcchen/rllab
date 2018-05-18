import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np


def _fc(x,
        output_dim,
        weight_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
        bias_initializer=tf.zeros_initializer):
    return tf.contrib.layers.fully_connected(x, output_dim, activation_fn=None, \
            weights_initializer=weight_initializer, weights_regularizer=None, biases_initializer=bias_initializer)


def _norm(x):
    return tc.layers.layer_norm(x, center=True, scale=True)


class ActorCriticBaseModel(object):
    def __init__(self, name):
        self.name = name

    @property
    def trainable_vars(self):
        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def setup_target_net(self, target_net, tau):
        vars = self.trainable_vars
        target_vars = target_net.trainable_vars

        soft_updates = []
        vars = vars[0:len(target_vars)]
        for var, target_var in zip(vars, target_vars):
            soft_updates.append(
                tf.assign(target_var, (1. - tau) * target_var + tau * var))
        assert len(soft_updates) == len(vars)
        self._update_parameters_op = tf.group(*soft_updates)


class ActorNet(ActorCriticBaseModel):
    def __init__(self, action_dim, lr=1e-4, bound=1, hidden_layers=[64, 64]):
        '''
        :param action_dim: the dimension of output action
        :param lr: learning rate 
        :param bound: the range of output action value
        :param hidden units in each layers
        '''
        self.name = 'actor'
        self._action_dim = action_dim
        self._lr = lr
        self._bound = bound
        self._hidden_layers = hidden_layers

    def build_net(self, state):
        self._state = state
        self._action = self._build_net(state) * self._bound

        with tf.variable_scope(self.name) as scope:
            self._optimizer = tf.train.AdamOptimizer(
                self._lr, beta1=0.9, beta2=0.999, epsilon=1e-08)

    def set_grad(self, grad):
        grads = tf.gradients(
            ys=self._action, xs=self.trainable_vars, grad_ys=grad)
        self._train_op = self._optimizer.apply_gradients(
            zip(grads, self.trainable_vars))

    def _build_net(self, state):

        with tf.variable_scope(self.name) as scope:
            fc_out = state
            for i, dim in enumerate(self._hidden_layers):
                with tf.variable_scope('fc_%d' % i):
                    fc_out = _fc(fc_out, dim)
                    fc_out = _norm(fc_out)
                    fc_out = tf.nn.relu(fc_out)

            with tf.variable_scope('output'):
                output = _fc(fc_out, self._action_dim)
                output = tf.tanh(output)
        return output

    def predict(self, session, state):
        return session.run(
            self.action, feed_dict={
                self._state: state,
            })

    def train(self, session, state):
        return session.run(
            [self._train_op], feed_dict={
                self._state: state,
            })

    def update_target_net(self, session):
        session.run(self._update_parameters_op)

    @property
    def action(self):
        return self._action


class CriticNet(ActorCriticBaseModel):
    def __init__(self,
                 weight_decay=0.01,
                 gamma=0.99,
                 lr=1e-3,
                 hidden_layers=[64, 64]):
        '''
        :param gamma: a discount factor 
        :param lr: learning rate 
        :param critic_l2_weight_decay: L2 weight decay 
        :param hidden units in each layers
        '''

        self.name = 'critic'

        self._weight_decay = weight_decay
        self._gamma = gamma
        self._lr = lr
        self._hidden_layers = hidden_layers

    def build_net(self, state, action, reward, terminal, target_q,
                  action_predict):
        self._state = state
        self._action = action
        self._reward = reward
        self._target_q = target_q
        self._terminal = terminal
        self._action_predict = action_predict

        self._q_value = self._build_net(state, action)

        #for calculating grade of actor, no meaning
        self._q_value_predict = self._build_net(state, action_predict, True)
        self._action_loss = -tf.reduce_mean(self._q_value_predict)

        self._get_target_Q(reward, terminal)
        self._get_loss(self._target_q)

    def _get_target_Q(self, rewards, terminals):
        self._target_Q = rewards + (
            1. - terminals) * self._gamma * self._q_value

    def _get_loss(self, target_q):
        self._loss = tf.reduce_mean(tf.square(target_q - self._q_value))

        critic_reg_vars = [
            var for var in self.trainable_vars
            if 'kernel' in var.name and 'output' not in var.name
        ]

        if self._weight_decay > 0.:
            vars = [
                var for var in self.trainable_vars
                if 'weights' in var.name and 'output' not in var.name
            ]
            vars += [
                var for var in self.trainable_vars
                if 'kernel' in var.name and 'output' not in var.name
            ]
            self._loss += tc.layers.apply_regularization(
                tc.layers.l2_regularizer(self._weight_decay),
                weights_list=vars)

        self._optimizer = tf.train.AdamOptimizer(
            self._lr, beta1=0.9, beta2=0.999, epsilon=1e-08)

        grads = tf.gradients(self._loss, self.trainable_vars)
        grads_and_vars = list(zip(grads, self.trainable_vars))
        self._train_op = self._optimizer.apply_gradients(grads_and_vars)

        self._action_grads = tf.gradients(self._action_loss,
                                          self._action_predict)

    def _build_net(self, state, action, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            fc_out = state
            for i, dim in enumerate(self._hidden_layers):
                with tf.variable_scope('fc_%d' % i):
                    if i == len(self._hidden_layers) - 1:
                        fc_out = tf.concat([fc_out, action], -1)
                    fc_out = _fc(fc_out, dim)
                    fc_out = _norm(fc_out)
                    fc_out = tf.nn.relu(fc_out)

            with tf.variable_scope('output'):
                output = _fc(
                    fc_out,
                    1,
                    weight_initializer=tf.random_uniform_initializer(
                        minval=-3e-3, maxval=3e-3))

        return output

    def predict(self, session, state, action):
        return session.run(
            self._q_value,
            feed_dict={
                self._state: state,
                self._action: action,
            })

    def predict_target_Q(self, session, state, action, reward, termial):
        return session.run(
            self._target_Q,
            feed_dict={
                self._state: state,
                self._action: action,
                self._reward: reward,
                self._terminal: termial,
            })

    def train(self, session, state, action, target_q):
        return session.run(
            [self._loss, self._train_op],
            feed_dict={
                self._state: state,
                self._action: action,
                self._target_q: target_q,
            })

    def update_target_net(self, session):
        session.run(self._update_parameters_op)

    def action_loss(self, session, state, action_predict):
        return session.run(
            self._action_loss,
            feed_dict={
                self._state: state,
                self._action_predict: action_predict,
            })

    @property
    def action_grads(self):
        return self._action_grads
