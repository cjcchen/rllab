import tensorflow as tf
import tensorflow.contrib as tc
import numpy as np
#def _fc(inputs, output_size, activation_fn=None, weights_initializer=tf.truncated_normal_initializer(),\
#        weights_regularizer=tf.contrib.layers.l2_regularizer(0.00), biases_initializer=tf.constant_initializer(0.0)):
#    return tf.contrib.layers.fully_connected(inputs, output_size, activation_fn=activation_fn, \
#            weights_initializer=weights_initializer, weights_regularizer=weights_regularizer, biases_initializer=biases_initializer)

def _fc(x, output_dim, 
        weight_initializer = tf.truncated_normal_initializer(stddev=0.01), 
        bias_initializer = tf.zeros_initializer
      ):

    weights = tf.get_variable('weights', shape=[x.shape[-1], output_dim], initializer=weight_initializer)
    biases = tf.get_variable('biases', shape=[output_dim], initializer=tf.zeros_initializer)
    return tf.nn.xw_plus_b(x, weights, biases)

def _bn(x,training_phase,scope_bn,activation=None):
    return tf.cond(training_phase, 
        lambda: tf.contrib.layers.batch_norm(x, activation_fn=activation, center=True, scale=True,
            updates_collections=None,is_training=True, reuse=None,scope=scope_bn,decay=0.9, epsilon=1e-5),
        lambda: tf.contrib.layers.batch_norm(x, activation_fn =activation, center=True, scale=True,
            updates_collections=None,is_training=False, reuse=True,scope=scope_bn,decay=0.9, epsilon=1e-5))

def _get_trainable_variables(name):
    variables = tf.trainable_variables()
    res=[]
    for v in variables:
        if v.name.find(name)>=0:
            print ("get var:",v.name)
            res.append(v)
    print("len:",len(res))
    return res

def _get_summary_op(name_id):

    summary_op = []
    scale_value = tf.get_collection("scale_summary")
    for (name, var) in scale_value:
        if var.op.name.find(name_id)>=0:
            summary_op.append(tf.summary.scalar(name, var))

    histogram_value = tf.get_collection("histogram_summary")
    for var in histogram_value:
        if isinstance( var, list):
            for v in var:
                print (v)
                if v.op.name.find(name_id)>=0:
                    print ("get histogram:",v.op.name)
                    summary_op.append(tf.summary.histogram(v.op.name, v))
        else:
            print (var)
            if var.op.name.find(name_id)>=0:
                print ("get histogram:",var.op.name)
                summary_op.append(tf.summary.histogram(var.op.name, var))
    
    return summary_op

class Model(object):
    def __init__(self, name):
        self.name = name
    

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]

    def setup_target_net(self, target_net, tau):
        vars = self.trainable_vars
        target_vars = target_net.trainable_vars

        soft_updates = []
        vars = vars[0: len(target_vars)]
        print ("var:",vars)
        print ("tar var:",target_vars)
        print (len(vars), len(target_vars))

        for var, target_var in zip(vars, target_vars):
            soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
        assert len(soft_updates) == len(vars)
        self._update_paras = tf.group(*soft_updates)



class ActorNet(Model):
    def __init__(self, sess, action_dim,lr = 1e-4, bound = 1):
        self.name='actor'
        self._action_dim = action_dim
        self._lr = lr
        self._session = sess
        self._bound = bound

    def build_net(self, state):
        self._state = state
        self._action = self._build_net(state)*self._bound

        with tf.variable_scope(self.name) as scope:
            self._optimizer = tf.train.AdamOptimizer(self._lr,beta1=0.9, beta2=0.999, epsilon=1e-08)

            #self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])
            #self.unnormalized_actor_gradients = tf.gradients(self.action, self.trainable_vars, -self.action_gradient)
            #self.actor_gradients = list(map(lambda x: tf.div(x, 64), self.unnormalized_actor_gradients))

            #self.optimize = tf.train.AdamOptimizer(1e-4).apply_gradients(zip(self.actor_gradients, self.trainable_vars))


    def _build_net(self, state):

        with tf.variable_scope(self.name) as scope:
            x = state
            x = tf.layers.dense(x, 64)
            tf.add_to_collection('histogram_summary',x);
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 64)
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self._action_dim, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)

        return x
    
    def predict(self, state):
        #print ("predict:",self._state, state.shape)
        return self._session.run(self.action, feed_dict = {
            self._state:state, 
        })


    def train(self, state, grads):
        return self._session.run([self._optimize], feed_dict={
            self._state: state,
            self._action_gradients: grads,
            })

    def update_target_net(self):
        self._session.run(self._update_paras)

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def action(self):
        return self._action

class CriticNet(Model):
    def __init__(self, sess, weight_decay = 0.01, gamma = 0.99, lr = 1e-3):
        self.name='critic'

        self._weight_decay = weight_decay
        self._gamma = gamma
        self._lr = lr
        self._session = sess
    
    def build_net(self, state, action, reward, terminal, target_q, action_predict):
        self._state = state
        self._action = action
        self._reward = reward
        self._target_q = target_q
        self._terminal = terminal
        
        self._q_value = self._build_net(state, action)
        #for calculating grade of actor, no meaning
        self._q_value_predict = self._build_net(state, action_predict, True)
        
        self._get_target_Q(reward, terminal)
        self._get_loss(self._target_q)
        self._action_loss = -tf.reduce_mean(self._q_value_predict)

    def _get_target_Q(self, rewards, terminals):
        self._target_Q = rewards + (1. - terminals) * self._gamma * self._q_value_predict

    def _get_loss(self, target_q):
        self._loss = tf.reduce_mean(tf.square(target_q - self._q_value))

        critic_reg_vars = [var for var in self.trainable_vars if 'kernel' in var.name and 'output' not in var.name]

        if self._weight_decay > 0.:
            vars = [var for var in self.trainable_vars if 'kernel' in var.name and 'output' not in var.name]
            self._loss += tc.layers.apply_regularization(tc.layers.l2_regularizer(self._weight_decay), weights_list=vars)

        self._optimizer = tf.train.AdamOptimizer(self._lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
        self._action_grads = tf.gradients(self._loss, self._action)


    def _build_net(self, state, action, reuse=False):

        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = state
            x = tf.layers.dense(x, 64)
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.concat([x, action], axis=-1)
            x = tf.layers.dense(x, 64)
            x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            output = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

        return output
    
    def predict(self, state, action):
        return self._session.run(self._q_value, feed_dict = {
                    self._state:state, 
                    self._action:action, 
                    })

    def predict_target_Q(self, state, reward, termial):
        return self._session.run(self.target_Q, feed_dict={
            self._state: state,
            self._reward: reward,
            self._terminal: termial,
        })


    def train(self, state, action, labels):
        return self._session.run([self._loss, self._optimize, self.action_grads], feed_dict={
            self._state: state,
            self._action: action,
            self._labels: labels,
            })

    def action_gradients(self, state, action):
        return self._session.run(self._action_grads, feed_dict={
            self._state: state,
            self._action: action,
        })

    def update_target_net(self):
        self._session.run(self._update_paras)

    @property
    def target_Q(self):
        return self._target_Q

    @property
    def optimizer(self):
        return self._optimizer
    
    @property
    def action_grads(self):
        return self._action_grads

    @property
    def loss(self):
        return self._loss

    @property
    def action_loss(self):
        return self._action_loss
