import torch
import torch.nn as nn
import cvxpy as cp
import numpy as np
import os.path
from SA_MDP_env import SAMDP_env
import safety_gymnasium
from stable_baselines3 import PPO, A2C, SAC, TD3
import torch
from fsrl.agent import PPOLagAgent
from fsrl.utils import TensorboardLogger
import numpy as np
import matplotlib.pyplot as plt
import json

def estimate_lower_bound(policy_gradient_func, state, perturbed_state, num_actions=10):
    """
    Estimate the lower bound (L') for a RL control policy using total variation distance and convexity assumptions.

    Args:
        policy_gradient_func (callable): A function that takes a state and returns the gradient of the policy with respect to that state.
        state (numpy.ndarray): The original state (s).
        perturbed_state (numpy.ndarray): The perturbed state (\tilde{s}).
        num_actions (int): Number of actions in the action space.

    Returns:
        float: Estimated lower bound (L').
    """
    # Calculate the gradient of the policy at the original state
    grad_pi_s = policy_gradient_func(state).cpu().squeeze(0)
    # print(grad_pi_s.shape)

    # Calculate the difference between perturbed state and original state
    delta_s = perturbed_state - state
    # print(delta_s.shape)
    # Calculate the linear approximation using convexity
    linear_approximation = grad_pi_s.T @ delta_s
    # print(linear_approximation)
    # Estimate the lower bound L'
    # lower_bound = np.min(linear_approximation)

    # Ensure the lower bound is strictly greater than 0
    lower_bound = max(abs(linear_approximation), 1e-5)

    return lower_bound

def estimate_overall_lower_bound(policy_gradient_func, state_space, num_actions=10, epsilon=None):
    """
    Estimate the overall lower bound (L) for all states in the state space.

    Args:
        policy_gradient_func (callable): A function that takes a state and returns the gradient of the policy with respect to that state.
        state_space (list of numpy.ndarray): A list containing all the states in the state space.
        num_actions (int): Number of actions in the action space.

    Returns:
        float: Estimated overall lower bound (L).
    """
    lower_bounds = []
    for i, state in enumerate(state_space):
        # Perturb the state slightly to calculate the lower bound for each state
        perturbed_state = state + np.random.normal(-0.1, 0.1, size=state.shape)
        lower_bound = estimate_lower_bound(policy_gradient_func, state, perturbed_state, num_actions)
        lower_bounds.append(lower_bound)

    # Calculate the overall lower bound as the minimum of all lower bounds
    overall_lower_bound = min((lower_bounds))

    return overall_lower_bound

# Example usage
def example_policy_gradient(observation):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)  # Shape: [1, obs_size]
    obs_tensor.requires_grad = True

    with torch.enable_grad():
        # action = model.policy.forward(obs_tensor)[0]
        #
        # # Sample action (or take the mean action)
        # # action = distribution.get_actions()
        # # log_prob = distribution.log_prob(action)
        #
        # # Calculate loss as negative log probability of the selected action
        # loss = -action.sum()
        # # Backward pass to compute gradients
        # model.policy.optimizer.zero_grad()
        # loss.backward(retain_graph=True)

        mean, log_std = model.actor(obs_tensor)[0]  # Get mean and log_std from the actor network
        # print(mean)
        # std = torch.exp(log_std)  # Convert log_std to std
        # dist = distributions.Normal(mean, std)
        # action = dist.sample()  # Sample an action from the distribution
        # log_prob = dist.log_prob(action)
        # action = distribution.get_actions()
        # log_prob = distribution.log_prob(action)

        # Calculate loss as negative log probability of the selected action
        # loss = -log_prob.sum()
        # Backward pass to compute gradients
        loss = -mean.sum()
        model.optim.zero_grad()
        loss.backward()

    # print(obs_tensor.grad.data)
    # Collect the sign of the gradients
    # sign_data_grad = obs_tensor.grad.data.sign()
    return obs_tensor.grad



level = 1
epsilon_list = [ 0.01, 0.03, 0.05, 0.07, 0.10]

render_mode = None
task_list = ['Goal', 'Button', 'Push','Race']
alg_name = PPO.__name__
alg_name = PPOLagAgent.__name__
for task in task_list:
    state_space = []

    file = f'./model/SafetyPoint{task}{level}-{alg_name}.pth'

    if not os.path.exists(file):
        print("File not exists!")
        continue
    # victim_model = PPO.load(file)
    env_id = f'SafetyPoint{task}{level}-v0'

    env = safety_gymnasium.make(env_id, render_mode=render_mode)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]  # Example output dimension
    # model = PPO.load(f'./model/SafetyPoint{task}{level}-{alg_name}.zip')
    # policy_net = model.policy
    agent = PPOLagAgent(env)
    agent.policy.load_state_dict(torch.load(f"./model/SafetyPoint{task}{level}-{alg_name}.pth"))
    model = agent.policy

    total_eposide = 1
    eposide = 0
    obs, info = env.reset()
    while eposide < total_eposide:
        action, _ = model.predict(obs)
        obs, reward, done, trun, info = env.step(action)
        state_space.append(np.array(obs))
        if done or trun:
            eposide += 1



    L = estimate_overall_lower_bound(example_policy_gradient, state_space)
    if L is not None:
        print(f"Estimated lower Lipschitz Constant for {task}{level}: {L}")
    else:
        print("Could not estimate Lipschitz constant.")

