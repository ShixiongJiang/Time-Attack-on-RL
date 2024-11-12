import random

import gymnasium
import safety_gymnasium

import numpy as np
import math
from copy import deepcopy

import torch
from stable_baselines3 import A2C, PPO



class SAMDP_safety_bench_env(gymnasium.Env):
    def __init__(self, victim_model=None,epsilon=0.5, env_id=None, config=None, seed=None, render_mode='human', semantics_method_tradition=False):
        # super(SafetyPointGoal1, self).__init__()
        self.total_time = 300
        self.safe_dis = 0.4
        self.obs = None
        self.hazard_dist = None
        self.goal_dist = None
        self.epsilon = epsilon
        if env_id is None:
            assert 'Please specify env_id'
        env_id = env_id
        safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode=render_mode)
        # self.env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
        self.env = safety_gymnasium_env

        self.observation_space = self.env.observation_space
        state_dim = self.env.observation_space.shape[0]

        pertub_constraint_los = np.ones(shape=state_dim) * self.epsilon * -1
        pertub_constraint_hig = np.ones(shape=state_dim) * self.epsilon
        self.action_space = gymnasium.spaces.Box(low=pertub_constraint_los, high=pertub_constraint_hig, dtype=np.float32)
        self.reward_cache = []
        self.avoid_reward_cache = []
        self.final_reward_cache = []
        self.steps = 0
        self.done = False
        self.seed = seed
        self.backdoor_trigger = False
        self.last_state = None
        self.victim_model = victim_model
        if self.victim_model == None:
            assert('please assign the victim policy')

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        obs, info = self.env.reset(seed=self.seed)
        self.steps = 0
        self.done = True
        self.obs = obs
        return obs, info


    def step(self, pertub_obs):

        self.last_obs = self.obs + pertub_obs
        # self.last_obs = self.obs
        # print(self.last_obs)
        self.steps += 1
        # print(self.last_obs)
        tensor = torch.from_numpy(self.last_obs)

        # Add a new dimension to make it of shape [1, 60]
        tensor = tensor.unsqueeze(0)
        action, _state = self.victim_model.actor.forward(tensor)
        action = action[0].squeeze().detach().numpy()

        obs, rew, cost, done, truncated, info = self.env.step(action)
        info['cost'] = info['cost_hazards']
        self.obs = obs

        adv_rew = -rew



        return obs, adv_rew, done, truncated, info



    def set_state(self, state):
        self.env = deepcopy(state)
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def get_state(self):
        return deepcopy(self.env)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()



class SAMDP_env(gymnasium.Env):
    def __init__(self, victim_model=None,epsilon=0.5, config=None, seed=None, render_mode='human', env_id=None, level=0):
        # super(SafetyPointGoal1, self).__init__()
        self.total_time = 300
        self.safe_dis = 0.4
        self.obs = None
        self.hazard_dist = None
        self.goal_dist = None
        self.epsilon = epsilon
        if env_id is None:
            assert 'please specify env_id'
        env_id = env_id
        safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode=render_mode)
        self.env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
        # self.env = safety_gymnasium_env
        self.observation_space = self.env.observation_space
        state_dim = self.env.observation_space.shape[0]

        pertub_constraint_los = np.ones(shape=state_dim) * self.epsilon * -1
        pertub_constraint_hig = np.ones(shape=state_dim) * self.epsilon
        self.action_space = gymnasium.spaces.Box(low=pertub_constraint_los, high=pertub_constraint_hig, dtype=np.float32)
        # print(type(self.action_space))
        self.radius = 0.2
        self.reward_cache = []
        self.avoid_reward_cache = []
        self.final_reward_cache = []
        self.steps = 0
        self.done = False
        self.seed = seed
        self.backdoor_trigger = False
        self.last_state = None
        self.victim_model = victim_model
        self.level = level
        if self.victim_model == None:
            assert('please assign the victim policy')

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        obs, info = self.env.reset(seed=self.seed)
        self.steps = 0
        self.done = True
        self.obs = obs
        return obs, info


    def step(self, pertub_obs):

        self.last_obs = self.obs + pertub_obs
        # self.last_obs = self.obs
        # print(self.last_obs)
        self.steps += 1
        # print(self.last_obs)
        if self.level == 1:
            tensor = torch.from_numpy(self.last_obs)

            tensor = tensor.unsqueeze(0)
            action, _state = self.victim_model.actor.forward(tensor)
            action = action[0].squeeze().detach().numpy()
            obs, rew, done, truncated, info = self.env.step(action)
            info['cost'] = info['cost_hazards']

        elif self.level == 0:
            action, _ = self.victim_model.predict(self.last_obs)

            obs, rew, done, truncated, info = self.env.step(action)
        # info['cost'] = info['cost_hazards']
        self.obs = obs

        adv_rew = -rew



        return obs, adv_rew, done, truncated, info



    def set_state(self, state):
        self.env = deepcopy(state)
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def get_state(self):
        return deepcopy(self.env)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
