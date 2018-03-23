import pygame
import numpy as np
from PIL import Image
from dm_control import suite
from dm_control.rl.environment import StepType
from dm_control.suite.wrappers import pixels
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete


'''
This environment will use dm_control toolkit to train and simulate your model.
Domain name and task name used for dm_control is showed below: (domain name/task name)

ball_in_cup/catch
cartpole/swingup_sparse
cartpole/balance_sparse
cartpole/swingup
cartpole/balance
cheetah/run
finger/turn_hard
finger/spin
finger/turn_easy
fish/upright
fish/swim
hopper/stand
hopper/hop
humanoid/stand
humanoid/run
humanoid/walk
manipulator/bring_ball
pendulum/swingup
point_mass/easy
reacher/hard
reacher/easy
swimmer/swimmer6
swimmer/swimmer15
walker/stand
walker/run
walker/walk
'''


def flat_shape(observation):
    dim = 0
    for key, value in observation.items():
        dim += int(np.prod(value.shape))
    return dim

def flat_observation(observation):
    new_observation=[]
    for key, value in observation.items():
        new_observation += value.flatten().tolist()
    observation = np.array(new_observation)
    return observation


class DMControlEnv(Env, Serializable):
    def __init__(self, domain_name, task_name = "swingup", 
            display = True, width = 320, height = 240, pixel_key = 'rbg', pixels_only = False, *arg, **kwargs):
        Serializable.quick_init(self, locals())
        pygame.init()

        #create pygame window
        self.display_window = pygame.display.set_mode((width,height))

        #create dm control env
        self.env = suite.load(domain_name=domain_name, task_name=task_name)
        self.wrapped = pixels.Wrapper(self.env,
                      observation_key=pixel_key,
                      pixels_only=pixels_only,
                      render_kwargs={'width': width, 'height': height})

        self.pixel_key = pixel_key
        self._display = display
        self._observation_spec = self.wrapped.observation_spec()
        self._action_spec = self.wrapped.action_spec()
        self._total_reward=0

    def step(self, action):
        time_step = self.wrapped.step(action)
        if self.display == True:
            print (time_step)
            pixels_img=time_step.observation[self.pixel_key]
            self._set_image(pixels_img)
        self._total_reward += time_step.reward
        
        return flat_observation(time_step.observation), time_step.reward, \
                1 if time_step.step_type == StepType.LAST else 0, \
                time_step.observation

    def reset(self):
        self._total_reward = 0
        time_step  = self.wrapped.reset()
        return flat_observation(time_step.observation)

    def render(self):
        if self.display == True:
            pygame.display.update()

    def _set_image(self,pixels_img):
        image = Image.fromarray(pixels_img)
        mode = image.mode
        size = image.size
        pygame_image = pygame.image.frombuffer(image.tobytes(), size, mode)
        self.display_window.blit(pygame_image, (0,0))
        pygame.display.update()

    def terminate(self):
        pygame.quit()

    @property
    def display(self):
        return self._display

    @property
    def action_space(self):
        return Box(self._action_spec.minimum, self._action_spec.maximum)

    @property
    def observation_space(self):
        flat_dim=flat_shape(self._observation_spec) 
        return Box(low=-np.inf, high=np.inf, shape=[flat_dim])

    @property
    def total_reward(self):
        return self._total_reward

