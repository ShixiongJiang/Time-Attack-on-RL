from stable_baselines3 import PPO
import torch as th
from env import SafetyPointGoal1_time, SafetyPointGoal1_delay_withoutTime

# env = SafetyPointGoal1_time(render_mode=None)
env = SafetyPointGoal1_delay_withoutTime(render_mode=None)

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[128, 64]))
# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_TimeState/")

model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_delayTime/")



# model.learn(total_timesteps=100000, tb_log_name="first_run_TimeState"
model.learn(total_timesteps=1000000, tb_log_name="first_run_delayTime_withoutTime")


# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_TimeState/")
#
# model.learn(total_timesteps=100000, tb_log_name="second_run_TimeState")
#
# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_TimeState/")
#
# model.learn(total_timesteps=100000, tb_log_name="third_run_TimeState")

model.save('SafetyPointGoal0-PPO-delayTime_withoutTim.zip')
