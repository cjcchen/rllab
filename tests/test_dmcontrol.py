from rllab.envs.dmcontrol_env import DMControlEnv
from rllab.envs.normalized_env import normalize
import numpy as np


model_list = [('ball_in_cup','catch'), ("cartpole","swingup_sparse"), ("cartpole","balance_sparse"), ("cartpole","swingup"), ("cartpole","balance"), ("cheetah","run"),  ("finger","spin"),("hopper","stand"), ("hopper","hop"), ("humanoid","stand"), ("humanoid","run"), ("humanoid","walk"), ("manipulator","bring_ball"), ("pendulum","swingup"), ("point_mass","easy"), ("reacher","hard"), ("reacher","easy"), ("swimmer","swimmer6"), ("swimmer","swimmer15"), ("walker","stand"), ("walker","run"), ("walker","walk")] 



def run_task(domain_name, task_name):
    try:
        print ("run: domain %s task %s"%(domain_name, task_name))
        dmcontrol_env = normalize(DMControlEnv(domain_name=domain_name, 
                            task_name=task_name, width=600, height = 400), normalize_obs=False, normalize_reward=False)

        time_step = dmcontrol_env.reset()
        action_spec = dmcontrol_env.action_space
        i = 0
        done = 0
        while done == 0:
            action = action_spec.sample()

            next_obs, reward, done, info = dmcontrol_env.step(action)
            #print(reward, done, next_obs, info)
            i+=1
            if i > 100:
                break

        dmcontrol_env.terminate()
        return 0
    except:
        return 1

for domain, task in  model_list:
    fail_list = []
    ret = run_task(domain, task)
    if ret == 1:
        fail_list.append((domain,task))

    print ("fail:",fail_list)


#run_task("humanoid","stand")
