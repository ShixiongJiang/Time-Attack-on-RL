import safety_gymnasium

from stable_baselines3 import PPO, A2C, SAC, TD3
from SA_MDP_env import SAMDP_safety_bench_env
import os
import torch as th

import torch
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from fsrl.agent import PPOLagAgent
from fsrl.utils import TensorboardLogger

level = 1
task_list = ['Goal', 'Push', 'Button', 'Race']
# alg_list = [PPO, A2C, SAC, TD3]
alg_list = [PPOLagAgent]
render_mode = None

# print(alg_list[0].__name__)

for task in task_list:
    for alg in alg_list:
        alg_name = alg.__name__

        logger = TensorboardLogger("logs", log_txt=True, name=task)

        file = f"./model/SafetyPoint{task}{level}-{alg_name}.pth"

        if os.path.exists(file):
            print("File exists!")
            continue

        env_id = f'SafetyPoint{task}{level}-v0'


        env = safety_gymnasium.make(env_id, render_mode=render_mode)
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)

        agent = PPOLagAgent(env, logger)
        agent.learn(env, env, epoch=50)
        torch.save(agent.policy.state_dict(), f"./model/SafetyPoint{task}{level}-{alg_name}.pth")



policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[128, 64]))
# train SA-MDP policy
for task in task_list:
    # for alg in alg_list:
        alg = PPOLagAgent
        env_id = f'SafetyPoint{task}{level}-v0'
        alg_name = alg.__name__
        logger = TensorboardLogger("logs", log_txt=True, name=task)

        file = f'./model/SAMDP_SafetyPoint{task}{level}-{alg_name}.pth'

        if os.path.exists(file):
            print("File exists!")
            continue

        victim_model = alg.load(f"./model/SafetyPoint{task}{level}-{alg_name}.pth")
        SAMDP_goal_env = SAMDP_safety_bench_env(env_id=env_id, render_mode=render_mode, victim_model=victim_model)



        agent = PPOLagAgent(SAMDP_goal_env, logger)
        agent.learn(SAMDP_goal_env, SAMDP_goal_env, epoch=50)
        torch.save(agent.policy.state_dict(), f"./model/SAMDP_SafetyPoint{task}{level}-{alg_name}.pth")






