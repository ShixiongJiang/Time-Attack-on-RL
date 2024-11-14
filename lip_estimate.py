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


def extract_network_parameters(policy_net):
    import torch.nn as nn
    weights = []
    biases = []
    activations = []

    # Function to extract parameters from a module
    def extract_from_module(module):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                weights.append(layer.weight.cpu().detach().numpy())
                biases.append(layer.bias.cpu().detach().numpy())
            elif isinstance(layer, nn.Conv2d):
                weights.append(layer.weight.cpu().detach().numpy())
                biases.append(layer.bias.cpu().detach().numpy())
            elif isinstance(layer, nn.ReLU):
                activations.append('relu')
            elif isinstance(layer, nn.Tanh):
                activations.append('tanh')
            # Add more activation functions if needed

    # Extract from features_extractor
    # extract_from_module(policy_net.features_extractor)
    # print(policy_net.action_net)
    # # Extract from mlp_extractor.policy_net
    # extract_from_module(policy_net.mlp_extractor.policy_net)
    #
    # # Extract from action_net (usually just a Linear layer)
    extract_from_module(policy_net.action_net)

    # Optionally, extract from mlp_extractor.value_net and value_net
    # extract_from_module(policy_net.mlp_extractor.value_net)
    # extract_from_module(policy_net.value_net)

    return weights, biases, activations


# LipSDP function
def lipsdp(weights, biases, activations):
    num_layers = len(weights)
    layer_dims = [w.shape[0] for w in weights]
    input_dim = weights[0].shape[1]

    # Define variables
    Ps = []
    for dim in layer_dims:
        P = cp.Variable((dim, dim), PSD=True)
        Ps.append(P)

    # Objective: Minimize the Lipschitz constant squared
    objective = cp.Minimize(cp.trace(Ps[-1]))

    # Constraints
    constraints = []
    # Initial constraint
    constraints.append(Ps[0] >> weights[0] @ weights[0].T)

    # Loop through layers
    for i in range(1, num_layers):
        Wi = weights[i]
        activation = activations[i-1]

        if activation == 'relu':
            Li = 1  # Lipschitz constant of ReLU is 1
        else:
            Li = 1  # Modify if other activations are used

        # Constraint: Ps[i] >= Wi * Ps[i-1] * Wi^T
        constraints.append(Ps[i] >> Wi @ Ps[i-1] @ Wi.T)

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)

    # Check if the problem was solved successfully
    if prob.status not in ["infeasible", "unbounded"]:
        L = np.sqrt(prob.value)
        return L
    else:
        print("Problem status:", prob.status)
        return None

level = 1

render_mode = None
task_list = ['Goal', 'Push', 'Button', 'Race']
alg_name = PPO.__name__
for task in task_list:
    file = f'./model/SafetyPoint{task}0-{alg_name}.zip'

    if not os.path.exists(file):
        print("File not exists!")
        continue
    victim_model = PPO.load(file)
    env_id = f'SafetyPoint{task}{level}-v0'

    env = safety_gymnasium.make(env_id, render_mode=render_mode)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]  # Example output dimension
    if level == 0:
        alg_name = PPO.__name__
        model = PPO.load(f'./model/SafetyPoint{task}{level}-{alg_name}.zip')
    else:
        alg_name = PPOLagAgent.__name__
        model = PPOLagAgent(env)
        model.policy.load_state_dict(torch.load(f"./model/SAMDP_SafetyPoint{task}{level}-{alg_name}.pth"))
        # model = agent.policy
    policy_net = model.policy
    # print(type(policy_net))
    weights, biases, activations = extract_network_parameters(policy_net)
    # Estimate the Lipschitz constant
    Lipschitz_constant = lipsdp(weights, biases, activations)
    if Lipschitz_constant is not None:
        print(f"Estimated Lipschitz Constant (LipSDP) for {task}{level}: {Lipschitz_constant}")
    else:
        print("Could not estimate Lipschitz constant.")







