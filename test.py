from stable_baselines3 import PPO
import torch
from SA_MDP_env import SAMDP_safetygoal1_env, safetygoal1_env
from fsrl.agent import PPOLagAgent
from fsrl.utils import TensorboardLogger
import numpy as np
import matplotlib.pyplot as plt
# env = SafetyPointGoal1_time(render_mode=None)


policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[128, 64], vf=[128, 64]))

# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_delayTime/")
base_env = safetygoal1_env(render_mode=None)


task = "Safetypointgoal-v1"
logger = TensorboardLogger("logs", log_txt=True, name=task)

agent_baseline = PPOLagAgent(base_env)
agent_baseline.policy.load_state_dict(torch.load('model/PPOLag_policy_for_pointgoal1_baseline.pth'))
# print(agent_baseline.policy.actor.forward())

env = SAMDP_safetygoal1_env(victim_model=agent_baseline.policy, render_mode='human')


# init the PPO Lag agent with default parameters
agent = PPOLagAgent(env, logger)
agent.policy.load_state_dict(torch.load('model/advPPOLag_policy_for_pointgoal1_baseline.pth'))

total_reward = 0
total_reach = 0
total_violate = 0
eposide = 0
epsilon = 0.1
obs, info = env.reset()
epsilon_list = [ 0.01, 0.3, 0.4]
total_eposide = 30
avg_time_list = []
cost_sum_list = []
for epsilon in epsilon_list:
    cost = 0
    reach_count = 1
    avg_time_total = 0
    eposide = 0

    while eposide < total_eposide:
        tensor = torch.from_numpy(obs)

        tensor = tensor.unsqueeze(0)
        action, _state = agent.policy.actor.forward(tensor)
        action = action[0].squeeze().detach().numpy()
        action = np.where(action > epsilon, epsilon, action)
        action = np.where(action < -epsilon, -epsilon, action)
        # print(action)
        obs, reward, done, trun, info = env.step(action)
        if 'goal_met' in info:
            if info['goal_met']:
                reach_count += 1
        # print(obs[12:28])
        cost += info['cost_hazards']
        if done or trun:
            eposide += 1
            avg_time_total += env.steps / reach_count
            reach_count = 1
            env.reset()

    # print(avg_time_total)
    avg_time_list.append(avg_time_total / total_eposide)
    cost_sum_list.append(cost / total_eposide)

plt.plot(epsilon_list, avg_time_list, marker='o', linestyle='-', color='b', label='Avg Time')
# plt.plot(epsilon_list, cost_sum_list, marker='o', linestyle='-', color='r', label='Avg cost')
plt.show()

# plt.plot(epsilon_list, avg_time_list, marker='o', linestyle='-', color='b', label='Avg Time')
plt.plot(epsilon_list, cost_sum_list, marker='o', linestyle='-', color='r', label='Avg cost')
plt.show()
