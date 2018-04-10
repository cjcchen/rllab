import tensorflow as tf
import numpy as np
import ddpg_config 
from src.ops import fully_connected, batch_norm

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
            res.append(v)
    return res

def _get_summary_op(name_id):

    summary_op = []
    scale_value = tf.get_collection("scale_summary")
    for (name, var) in scale_value:
        if var.op.name.find(name_id)>=0:
            print ("get scale:",var.op.name, name)
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

class ActorNet:
    def __init__(self, session, state_dim, action_dim, action_bound):
        self._session = session

        with tf.variable_scope('actor_online'):
            self._state, self._action, self._action_bound, self._is_training \
                            = self._build_net(state_dim, action_dim, action_bound)
            self._params = _get_trainable_variables('actor_online')

        with tf.variable_scope('actor_target'):
            self._target_state, self._target_action, self._target_action_bound, self._target_is_training \
                            = self._build_net(state_dim, action_dim, action_bound)
            self._target_params = _get_trainable_variables('actor_target')

        #copy params to target network
        self._update_target_params = []
        for i in range(len(self._target_params)):
            new_params = tf.multiply(self._params[i], ddpg_config.TAU) + tf.multiply(self._target_params[i], 1-ddpg_config.TAU)
            update_op = self._target_params[i].assign(new_params)
            self._update_target_params.append(update_op)

        with tf.variable_scope('actor_update'):
            #dJ/da
            self._action_gradients = tf.placeholder(tf.float32, [None, action_dim])
            #dJ/dw = dJ/da*da/dw
            actor_gradients = tf.gradients(self._action, self._params, -self._action_gradients)
            #tf.add_to_collection('histogram_summary',actor_gradients);

            self._optimize = tf.train.AdamOptimizer(ddpg_config.ACTOR_LEARNING_RATE).apply_gradients(zip(actor_gradients, self._params))

        self._summary_str_op = _get_summary_op('actor_online')
        self._summary_str_op += _get_summary_op('actor_update')
        self._summary_str_op = tf.summary.merge(self._summary_str_op)


    def _build_net(self, state_dim, action_dim, action_bound):
        state = tf.placeholder(tf.float32, shape=[None,state_dim], name="state")
        is_training = tf.placeholder(tf.bool, name="is_training")

        with tf.variable_scope('fc1'):
            fc1_out = _fc(state, 400)
            fc1_out = _bn(fc1_out, is_training, scope_bn="BN_0")
            fc1_out = tf.nn.relu(fc1_out)
            tf.add_to_collection('histogram_summary',fc1_out);

        with tf.variable_scope('fc2'):
            fc2_out = _fc(fc1_out, 300)
            fc2_out = _bn(fc2_out, is_training, scope_bn="BN_1")
            fc2_out = tf.nn.relu(fc2_out)
            tf.add_to_collection('histogram_summary',fc2_out);
       
        with tf.variable_scope('output'):
            output = _fc(fc2_out, action_dim)
            output = tf.tanh(output) 
            tf.add_to_collection('histogram_summary',output);

        scaled_outputs = tf.multiply(output, action_bound) 

        return state, output, scaled_outputs, is_training
    
    def predict(self, state):
        return self._session.run(self._action_bound, feed_dict = {
            self._state:state, 
            self._is_training:False
        })

    def predict_target(self, state):
        return self._session.run(self._target_action_bound, feed_dict = {
            self._target_state:state, 
            self._target_is_training:False
        })


    def train(self, state, grads):
        return self._session.run([self._summary_str_op,self._optimize], feed_dict={
            self._state: state,
            self._action_gradients: grads,
            self._is_training: True
            })

    def update_target_net(self):
        self._session.run(self._update_target_params)

class CriticNet:
    def __init__(self, session, state_dim, action_dim):
        self._session = session

        with tf.variable_scope('critic_online'):
            self._state, self._action, self._q_value, self._is_training = self._build_net(state_dim, action_dim)
            self._params = _get_trainable_variables('critic_online')

        with tf.variable_scope('critic_target'):
            self._target_state, self._target_action, self._target_q_value, self._target_is_training = self._build_net(state_dim, action_dim)
            self._target_params = _get_trainable_variables('critic_target')


        self._update_target_params = []
        for i in range(len(self._target_params)):
            new_params = tf.multiply(self._params[i], ddpg_config.TAU) + tf.multiply(self._target_params[i], 1-ddpg_config.TAU)
            update_op = self._target_params[i].assign(new_params)
            self._update_target_params.append(update_op)


        with tf.variable_scope('critic_update'):
            self._labels = tf.placeholder(tf.float32, [None, 1], name="labels")

            # L2 loss
            self._loss = tf.reduce_mean(tf.squared_difference(self._labels, self._q_value))
            self._optimize = tf.train.AdamOptimizer(ddpg_config.CRITIC_LEARNING_RATE).minimize(self._loss)

            tf.add_to_collection('scale_summary',('loss',self._loss));

            # used for calculate the gradient of action net
            self._action_grads = tf.gradients(self._q_value, self._action)
            tf.add_to_collection('histogram_summary',self._action_grads);

        self._summary_str_op = _get_summary_op('critic_online')
        self._summary_str_op += _get_summary_op('critic_update')
        self._summary_str_op = tf.summary.merge(self._summary_str_op)

    def _build_net(self, state_dim, action_dim):
        state = tf.placeholder(tf.float32, shape=[None,state_dim], name="states")
        action = tf.placeholder(tf.float32, shape=[None, action_dim], name="actions")

        is_training = tf.placeholder(tf.bool)

        with tf.variable_scope('fc1'):
            fc1_out = _fc(state, 400)
            #fc1_out = _bn(fc1_out, is_training, scope_bn="BN_0")
            fc1_out = tf.nn.relu(fc1_out)
            tf.add_to_collection('histogram_summary',fc1_out);
    
        with tf.variable_scope('fc2'):
            fc2_in = tf.concat([fc1_out, action], 1)
            fc2_out = _fc(fc2_in, 300)
            #fc2_out = _bn(fc2_out, is_training, scope_bn="BN_1")
            fc2_out = tf.nn.relu(fc2_out)
            tf.add_to_collection('histogram_summary',fc2_out);
       
        with tf.variable_scope('output'):
            output = _fc(fc2_out, 1)
            tf.add_to_collection('histogram_summary',output);

        return state, action, output, is_training
    
    def predict(self, state, action):
        return self._session.run(self._q_value, feed_dict = {
                    self._state:state, 
                    self._action:action, 
                    self._is_training:False
                    })

    def predict_target(self, state, action):
        return self._session.run(self._target_q_value, feed_dict = {
                    self._target_state:state, 
                    self._target_action:action, 
                    self._target_is_training:False
                    })


    def train(self, state, action, labels):
        return self._session.run([self._summary_str_op, self._loss, self._optimize], feed_dict={
            self._state: state,
            self._action: action,
            self._labels: labels,
            self._is_training: True
            })

    def action_gradients(self, state, action):
        return self._session.run(self._action_grads, feed_dict={
            self._state: state,
            self._action: action,
            self._is_training: False
        })


    def update_target_net(self):
        self._session.run(self._update_target_params)
