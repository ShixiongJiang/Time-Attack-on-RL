from stable_baselines3 import PPO
import torch
from SA_MDP_env import SAMDP_safetygoal1_env, safetygoal1_env
from fsrl.agent import PPOLagAgent
from fsrl.utils import TensorboardLogger
# env = SafetyPointGoal1_time(render_mode=None)


policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[128, 64]))

# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_delayTime/")
base_env = safetygoal1_env(render_mode=None)

# model = PPO("MlpPolicy", base_env, verbose=1, policy_kwargs=policy_kwargs)
#
# # model.learn(total_timesteps=100000, tb_log_name="first_run_TimeState"
# model = PPO.load('model/SafetyPointGoal1-PPO-baseline.zip', env=base_env)
task = "Safetypointgoal-v1"
logger = TensorboardLogger("logs", log_txt=True, name=task)

agent_baseline = PPOLagAgent(base_env)
agent_baseline.policy.load_state_dict(torch.load('model/PPOLag_policy_for_pointgoal1_baseline.pth'))
# print(agent_baseline.policy.actor.forward())

env = SAMDP_safetygoal1_env(victim_model=agent_baseline.policy, render_mode=None)


# init the PPO Lag agent with default parameters
agent = PPOLagAgent(env, logger)
agent.learn(env, env, epoch=50)
torch.save(agent.policy.state_dict(), "model/advPPOLag_policy_for_pointgoal1_baseline.pth")
# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_TimeState/")
#
# model.learn(total_timesteps=100000, tb_log_name="second_run_TimeState")
#
# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_TimeState/")
#
# model.learn(total_timesteps=100000, tb_log_name="third_run_TimeState")

