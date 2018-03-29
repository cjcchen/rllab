import numpy as np

from rllab.envs.dm_control_env import DmControlEnv
from rllab.envs.normalized_env import normalize

# yapf: disable
model_list = [('ball_in_cup', 'catch'),
           ("cartpole", "swingup_sparse"),
           ("cartpole", "balance_sparse"),
           ("cartpole", "swingup"),
           ("cartpole", "balance"),
           ("cheetah", "run"),
           ("finger", "spin"),
           ("hopper", "stand"),
           ("hopper", "hop"),
           ("humanoid", "stand"),
           ("humanoid", "run"),
           ("humanoid", "walk"),
           ("manipulator", "bring_ball"),
           ("pendulum", "swingup"),
           ("point_mass", "easy"),
           ("reacher", "hard"),
           ("reacher", "easy"),
           ("swimmer", "swimmer6"),
           ("swimmer", "swimmer15"),
           ("walker", "stand"),
           ("walker", "run"),
           ("walker", "walk")]
# yapf: enable


def run_task(domain_name, task_name):
    print("run: domain %s task %s" % (domain_name, task_name))
    dmcontrol_env = normalize(
        DmControlEnv(
            domain_name=domain_name,
            task_name=task_name,
            plot=True,
            width=600,
            height=400),
        normalize_obs=False,
        normalize_reward=False)

    time_step = dmcontrol_env.reset()
    action_spec = dmcontrol_env.action_space
    for i in range(100):
        dmcontrol_env.render()
        action = action_spec.sample()
        next_obs, reward, done, info = dmcontrol_env.step(action)
        if done == True:
            break

    dmcontrol_env.terminate()


for domain, task in model_list:
    run_task(domain, task)

print("Congratulation! All tasks are done!")
