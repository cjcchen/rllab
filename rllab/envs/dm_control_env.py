import pygame
import numpy as np

from dm_control import suite
from dm_control.rl.environment import StepType

from rllab.envs.base import Env, Step
from rllab.envs.dm_control_view import DmControlViewer
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
'''
This environment will use dm_control toolkit(https://arxiv.org/pdf/1801.00690.pdf) 
to train and simulate your models.
'''


def _flat_shape(observation):
    return np.sum( int(np.prod(v.shape)) for k,v in observation.items())


def _flat_observation(observation):
    return np.hstack(v.flatten() for k,v in observation.items())


class DmControlEnv(Env, Serializable):
    def __init__(self,
                 domain_name,
                 task_name,
                 plot=False,
                 width=320,
                 height=240,
                 ):
        Serializable.quick_init(self, locals())

        self._env = suite.load(domain_name=domain_name, task_name=task_name)

        self._total_reward = 0
        self._render_kwargs={ 'width': width, 'height': height }

        if plot:
            self._viewer = DmControlViewer()
        else:
            self._viewer = None


    def step(self, action):
        time_step = self._env.step(action)
        if time_step.reward:
            self._total_reward += time_step.reward

        return Step(_flat_observation(time_step.observation), time_step.reward, \
                True if time_step.step_type == StepType.LAST else False, \
                **time_step.observation)

    def reset(self):
        self._total_reward = 0
        time_step = self._env.reset()
        return _flat_observation(time_step.observation)

    def render(self):
        if self._viewer:
            pixels_img = self._env.physics.render(**self._render_kwargs)
            self._viewer.loop_once(pixels_img)

    def terminate(self):
        if self._viewer:
            self._viewer.finish()

    @property
    def action_space(self):
        action_spec = self._env.action_spec()
        return Box(action_spec.minimum, action_spec.maximum)

    @property
    def observation_space(self):
        flat_dim = _flat_shape(self._env.observation_spec())
        return Box(low=-np.inf, high=np.inf, shape=[flat_dim])

    @property
    def total_reward(self):
        return self._total_reward

