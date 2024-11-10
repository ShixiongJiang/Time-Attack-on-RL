import safety_gymnasium

from stable_baselines3 import PPO, A2C, SAC, TD3
from SA_MDP_env import SAMDP_env

import torch as th
from fsrl.agent import PPOLagAgent
from fsrl.utils import TensorboardLogger
import torch

task_list = ['Goal', 'Push', 'Button', 'Race']
alg_list = [PPO, A2C, SAC, TD3]
render_mode = None
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[128, 64]))
# print(alg_list[0].__name__)

for task in task_list:
    for alg in alg_list:
        env_id = f'SafetyPoint{task}0-v0'
        alg_name = alg.__name__
        env = safety_gymnasium.make(env_id, render_mode=render_mode)
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
        model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=f"./logs/{alg_name}_{task}0_env/")

        model.learn(total_timesteps=1000000, tb_log_name=f"{task}0_env_first")

        model.save(f'./model/SafetyPoint{task}0-{alg_name}.zip')

# train SA-MDP policy
for task in task_list:
    for alg in alg_list:
        env_id = f'SafetyPoint{task}0-v0'
        alg_name = alg.__name__
        victim_model = PPO.load(f'./model/SafetyPoint{task}0-{alg_name}.zip')
        SAMDP_goal_env = SAMDP_env(env_id=env_id, render_mode=render_mode, victim_model=victim_model)

        model = PPO("MlpPolicy", SAMDP_goal_env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=f"./logs/SAMDP_{alg_name}_{task}0_env/")

        model.learn(total_timesteps=1000000, tb_log_name=f"SAMDP_{task}0_env_first")

        model.save(f'./model/SAMDP_SafetyPoint{task}0-{alg_name}.zip')



