import numpy as np
import random

class ReplayBuffer:
    def __init__(self, buffer_size):
        self._buffer = []
        self._buffer_size = buffer_size

    def add_data(self, state, action,reward,terminal, next_state):
        self._buffer.append( (state, action, reward, terminal, next_state ) )
        if(self.get_buffer_size()>self._buffer_size):
            self._buffer = self._buffer[1:] 

    def get_batch_data(self, batch_size):
        data=random.sample(self._buffer, batch_size)
        states = np.array([d[0] for d in data])
        actions = np.array([d[1] for d in data]) 
        rewards = np.array([d[2] for d in data]) 
        terminals = np.array([d[3] for d in data]) 
        next_states = np.array([d[4] for d in data])
        return [states, actions, rewards, terminals, next_states]

    def get_buffer_size(self):
        return len(self._buffer)
