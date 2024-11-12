import os.path
from SA_MDP_env import  SAMDP_safety_bench_env

from stable_baselines3 import PPO, A2C, SAC, TD3
import torch
from fsrl.agent import PPOLagAgent
from fsrl.utils import TensorboardLogger
import numpy as np
import matplotlib.pyplot as plt
import json

import safety_gymnasium

# env = SafetyPointGoal1_time(render_mode=None)
if not os.path.exists('./figs'):
    os.makedirs('./figs')
# policy_kwargs = dict(activation_fn=torch.nn.ReLU,
#                      net_arch=dict(pi=[128, 64], vf=[128, 64]))
#
# # model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_delayTime/")
# base_env = safetygoal1_env(render_mode=None)
#
#
# task = "Safetypointgoal-v1"
# logger = TensorboardLogger("logs", log_txt=True, name=task)
#
# agent_baseline = PPOLagAgent(base_env)
# agent_baseline.policy.load_state_dict(torch.load('model/PPOLag_policy_for_pointgoal1_baseline.pth'))
# # print(agent_baseline.policy.actor.forward())
#
# env = SAMDP_safetygoal1_env(victim_model=agent_baseline.policy, render_mode='human')
#
#
# # init the PPO Lag agent with default parameters
# agent = PPOLagAgent(env, logger)
# agent.policy.load_state_dict(torch.load('model/advPPOLag_policy_for_pointgoal1_baseline.pth'))
def fgsm_attack(observation, epsilon, model):
    """
    Performs the FGSM attack on the given observation.

    Args:
        observation (numpy.ndarray): The original observation.
        epsilon (float): The perturbation magnitude.
        policy (nn.Module): The trained policy network.

    Returns:
        torch.Tensor: The perturbed observation.
    """
    # Convert observation to tensor and enable gradient
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)  # Shape: [1, obs_size]
    obs_tensor.requires_grad = True

    with torch.enable_grad():
        distribution = model.policy.get_distribution(obs_tensor)

        # Sample action (or take the mean action)
        action = distribution.get_actions()
        log_prob = distribution.log_prob(action)

        # Calculate loss as negative log probability of the selected action
        loss = -log_prob

        # Backward pass to compute gradients
        model.policy.optimizer.zero_grad()
        loss.backward()

    # Collect the sign of the gradients
    sign_data_grad = obs_tensor.grad.data.sign()

    # Create the perturbed observation
    perturbed_obs = obs_tensor + epsilon * sign_data_grad

    # Move perturbed observation back to CPU and convert to numpy
    perturbed_obs = perturbed_obs.detach().cpu().numpy()[0]
    return perturbed_obs
def Random(observation, epsilon):
    noise = np.random.uniform(low=-epsilon, high=epsilon, size=observation.shape)

    # Add noise to the observation
    perturbed_observation = observation + noise

    return perturbed_observation

level = 1
task_list = ['Goal', 'Push', 'Button', 'Race']
# task_list = ['Push']
adv_method_list = ['Optimal Time Attack', 'FGSM Attack', 'Random']
render_mode = 'human'
# policy_kwargs = dict(activation_fn=th.nn.ReLU,
#                      net_arch=dict(pi=[128, 64], vf=[128, 64]))
fig, ax = plt.subplots()
fig.set_figheight(4.6)
fig.set_figwidth(12)
plt.yticks(fontsize=36)
plt.xticks(fontsize=36)
plt.xlabel('Perturbation Range', fontsize='36')
plt.ylabel('Time',fontsize='36')
plt.legend(fontsize=26)


plt.tight_layout()  # Adjust layout to not cut off any labels or legends
for task in task_list:
    color_list = ['purple', "red", "green","orange"]
    color_iterator = iter(color_list)
    for adv_method in adv_method_list:
        alg_name = 'PPOLagAgent'
        env_id = f'SafetyPoint{task}{level}-v0'

        victim_file = f'./model/SafetyPoint{task}{level}-{alg_name}.pth'
        adv_file = f'./model/SAMDP_SafetyPoint{task}{level}-{alg_name}.pth'
        avg_time_list = []
        cost_sum_list = []
        avg_timd_std_list = []
        if not os.path.exists(victim_file) or not os.path.exists(adv_file):
            print("File not exists!")
            continue

        env = safety_gymnasium.make(env_id, render_mode=render_mode)
        env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
        victim_agent = PPOLagAgent(env)
        victim_agent.policy.load_state_dict(torch.load(f"./model/SafetyPoint{task}{level}-{alg_name}.pth"))
        victim_model = victim_agent

        SAMDP_goal_env = SAMDP_safety_bench_env(env_id=env_id, render_mode=render_mode, victim_model=victim_model)


        env = SAMDP_goal_env
        adv_agent = PPOLagAgent(env)
        adv_agent.policy.load_state_dict(torch.load(f"./model/SAMDP_SafetyPoint{task}{level}-{alg_name}.pth"))
        adv_model = adv_agent.policy

        total_reward = 0
        total_reach = 0
        total_violate = 0
        eposide = 0
        obs, info = env.reset()
        epsilon_list = [ 0.01, 0.03, 0.05, 0.07, 0.10]
        total_eposide = 50
        seed = [1,2,3,4,5]

        for epsilon in epsilon_list:
            cost = 0
            reach_count = 1
            avg_time_total = 0
            eposide = 0
            time_list = []
            while eposide < total_eposide:
                if adv_method == 'Optimal Time Attack':
                    tensor = torch.from_numpy(obs)
                    tensor = tensor.unsqueeze(0)
                    perturbed_obs, _state = adv_model.actor.forward(tensor)
                    perturbed_obs = perturbed_obs[0].squeeze().detach().numpy()
                elif adv_method == 'FGSM Attack':
                    perturbed_obs = fgsm_attack(obs, epsilon=epsilon, model=victim_model)
                elif adv_method == 'Random':
                    perturbed_obs = Random(observation=obs, epsilon=epsilon)
                # print(action)
                perturbed_obs = np.where(perturbed_obs > epsilon, epsilon, perturbed_obs)
                perturbed_obs = np.where(perturbed_obs < -epsilon, -epsilon, perturbed_obs)
                # print(action)
                obs, reward, done, trun, info = env.step(perturbed_obs)
                if 'goal_met' in info:
                    if info['goal_met']:
                        reach_count += 1
                # print(obs[12:28])
                # cost += info['cost_hazards']
                if done or trun:
                    eposide += 1
                    avg_time_total += env.steps / reach_count
                    time_list.append(env.steps / reach_count)
                    reach_count = 1
                    env.reset()

            # print(avg_time_total)
            avg_time_list.append(avg_time_total / total_eposide)
            avg_timd_std_list.append(np.std(time_list))
            # cost_sum_list.append(cost / total_eposide)

        color = next(color_iterator)
        # print(avg_time_list)
        # print(avg_timd_std_list)
        data = {
            'avg_time_list': avg_time_list,
            'avg_timd_std_list': avg_timd_std_list
        }

        # Save to JSON file
        with open(f'./figs/{task}{adv_method}{level}-data.json', 'w') as f:
            json.dump(data, f)
        plt.plot(epsilon_list, avg_time_list, color=color, label=f"Line {adv_method}")
        plt.fill_between(epsilon_list, np.array(avg_time_list) - np.array(avg_timd_std_list), np.array(avg_time_list) + np.array(avg_timd_std_list), color=color, alpha=0.2)

    # plt.xlabel("Perturbation range")
    # plt.ylabel("Y-axis")
    plt.title(f'{task}')
    plt.legend()
    # plt.show()
    plt.savefig(f'./figs/{task}.pdf',bbox_inches='tight', dpi=500)
# total_reward = 0
# total_reach = 0
# total_violate = 0
# eposide = 0
# epsilon = 0.1
# obs, info = env.reset()
# epsilon_list = [ 0.01, 0.04, 0.7, 0.10]
# total_eposide = 50
# seed = [1,2,3,4,5]
# avg_time_list = []
# cost_sum_list = []
# for epsilon in epsilon_list:
#     cost = 0
#     reach_count = 1
#     avg_time_total = 0
#     eposide = 0
#
#     while eposide < total_eposide:
#         tensor = torch.from_numpy(obs)
#
#         tensor = tensor.unsqueeze(0)
#         action, _state = agent.policy.actor.forward(tensor)
#         action = action[0].squeeze().detach().numpy()
#         action = np.where(action > epsilon, epsilon, action)
#         action = np.where(action < -epsilon, -epsilon, action)
#         # print(action)
#         obs, reward, done, trun, info = env.step(action)
#         if 'goal_met' in info:
#             if info['goal_met']:
#                 reach_count += 1
#         # print(obs[12:28])
#         cost += info['cost_hazards']
#         if done or trun:
#             eposide += 1
#             avg_time_total += env.steps / reach_count
#             reach_count = 1
#             env.reset()
#
#     # print(avg_time_total)
#     avg_time_list.append(avg_time_total / total_eposide)
#     cost_sum_list.append(cost / total_eposide)
#
# plt.plot(epsilon_list, avg_time_list, marker='o', linestyle='-', color='b', label='Avg Time')
# # plt.plot(epsilon_list, cost_sum_list, marker='o', linestyle='-', color='r', label='Avg cost')
# plt.show()
#
# # # plt.plot(epsilon_list, avg_time_list, marker='o', linestyle='-', color='b', label='Avg Time')
# # plt.plot(epsilon_list, cost_sum_list, marker='o', linestyle='-', color='r', label='Avg cost')
# plt.show()
