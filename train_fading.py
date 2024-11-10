from stable_baselines3 import PPO
import torch as th
from env import SafetyPointGoal1_time
from fading_env import fadingenv_time
# env = SafetyPointGoal1_time(render_mode=None)
env = fadingenv_time(render_mode=None)

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[128, 64]))
# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_TimeState/")

model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_fading_env_time/")



model.learn(total_timesteps=100000, tb_log_name="first_run_TimeState")
# model.learn(total_timesteps=1000000, tb_log_name="second_run_delayTime")


# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_TimeState/")
#
# model.learn(total_timesteps=100000, tb_log_name="second_run_TimeState")
#
# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_TimeState/")
#
# model.learn(total_timesteps=100000, tb_log_name="third_run_TimeState")

model.save('./model/SafetyPointGoal0-PPO-delayTime.zip')
