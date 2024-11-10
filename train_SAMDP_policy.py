import safety_gymnasium

from stable_baselines3 import PPO, A2C, SAC, TD3
import torch as th
from SA_MDP_env import SAMDP_env
from fsrl.agent import PPOLagAgent
from fsrl.utils import TensorboardLogger
import torch



goal_env_id = 'SafetyPointGoal0-v0'
render_mode = None
victim_model = PPO.load('./model/SafetyPointGoal0-PPO.zip')
SAMDP_goal_env = SAMDP_env(env_id=goal_env_id, render_mode=render_mode, victim_model=victim_model)


policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[128, 64]))

model = PPO("MlpPolicy", SAMDP_goal_env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_SAMDP_goal0_env/")

model.learn(total_timesteps=1000000, tb_log_name="SAMDP_goal0_env_first")

model.save('./model/SAMDP_SafetyPointGoal0-PPO.zip')



push_env_id = 'SafetyPointPush0-v0'
render_mode = None
victim_model = PPO.load('./model/SafetyPointPush0-PPO.zip')

SAMDP_push_env = SAMDP_env(env_id=push_env_id, render_mode=render_mode, victim_model=victim_model)


policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[128, 64]))

model = PPO("MlpPolicy", SAMDP_push_env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_SAMDP_push0_env/")

model.learn(total_timesteps=1000000, tb_log_name="SAMDP_push0_env_first")


button_env_id = 'SafetyPointButton0-v0'
victim_model = PPO.load('./model/SafetyPointbutton0-PPO.zip')

render_mode = None
SAMDP_button_env = SAMDP_env(env_id=button_env_id, render_mode=render_mode, victim_model=victim_model)


policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[128, 64]))

model = PPO("MlpPolicy", SAMDP_button_env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_SAMDP_button0_env/")

model.learn(total_timesteps=1000000, tb_log_name="SAMDP_button0_env_first")

