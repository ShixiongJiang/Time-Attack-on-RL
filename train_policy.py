import safety_gymnasium

from stable_baselines3 import PPO, A2C, SAC, TD3
from SA_MDP_env import SAMDP_env
import os
import torch as th

import torch
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise
from fsrl.agent import PPOLagAgent
from fsrl.utils import TensorboardLogger

level = 0
task_list = ['Goal', 'Push', 'Button', 'Race']
# alg_list = [PPO, A2C, SAC, TD3]
alg_list = [PPO]
render_mode = None
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[128, 64]))
# print(alg_list[0].__name__)

for task in task_list:
    for alg in alg_list:
        alg_name = alg.__name__


        file = f'./model/SafetyPoint{task}{level}-{alg_name}.zip'

        if os.path.exists(file):
            print("File exists!")
            continue

        env_id = f'SafetyPoint{task}{level}-v0'

        env = safety_gymnasium.make(env_id, render_mode=render_mode)
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
        if alg.__name__ == 'TD3':
            n_actions = env.action_space.shape[0]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
            policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64],  qf=[128, 64]))

            model = alg("MlpPolicy", env, policy_kwargs=policy_kwargs, action_noise=action_noise, verbose=0)


        elif alg.__name__ == 'A2C':
            n_actions = env.action_space.shape[0]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
            policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64]))

            model = alg("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0)
        else:
            model = alg("MlpPolicy", env, verbose=0, use_sde=True, tensorboard_log=f"./logs/{alg_name}_{task}{level}_env/")


        model.learn(total_timesteps=1000000, tb_log_name=f"{task}{level}_env_first")

        model.save(f'./model/SafetyPoint{task}{level}-{alg_name}.zip')


policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[128, 64]))
# train SA-MDP policy
for task in task_list:
    # for alg in alg_list:
        alg = PPO
        env_id = f'SafetyPoint{task}{level}-v0'
        alg_name = alg.__name__

        file = f'./model/SAMDP_SafetyPoint{task}{level}-{alg_name}.zip'

        if os.path.exists(file):
            print("File exists!")
            continue

        victim_model = alg.load(f'./model/SafetyPoint{task}{level}-{alg_name}.zip')
        SAMDP_goal_env = SAMDP_env(env_id=env_id, render_mode=render_mode, victim_model=victim_model)

        model = alg("MlpPolicy", SAMDP_goal_env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=f"./logs/SAMDP_{alg_name}_{task}{level}_env/")

        model.learn(total_timesteps=1000000, tb_log_name=f"SAMDP_{task}{level}_env_first")

        model.save(f'./model/SAMDP_SafetyPoint{task}{level}-{alg_name}.zip')



