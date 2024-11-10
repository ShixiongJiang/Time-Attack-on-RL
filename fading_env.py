import random

import gymnasium
import safety_gymnasium

import numpy as np
import math
from copy import deepcopy
from stable_baselines3 import A2C, PPO


# from gym import spaces

class fadingenv_time(gymnasium.Env):
    def __init__(self, config=None, seed=None, render_mode='human', semantics_method_tradition=False):
        # super(SafetyPointGoal1, self).__init__()
        self.total_time = 300
        self.safe_dis = 0.4
        self.obs = None
        self.hazard_dist = None
        self.goal_dist = None
        env_id = 'SafetyPointFadingHard1-v0'
        self.safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode=render_mode)
        self.env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(self.safety_gymnasium_env)
        # This default action sapce is wrong
        self.action_space = self.env.action_space
        # print(type(self.action_space))
        self.action_space = gymnasium.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        # print(type(self.action_space))
        self.observation_space = self.env.observation_space
        self.radius = 0.2
        self.reward_cache = []
        self.avoid_reward_cache = []
        self.final_reward_cache = []
        self.steps = 0
        self.done = False
        self.seed = seed
        self.backdoor_trigger = False
        self.last_state = None
        self.semantics_method_tradition=semantics_method_tradition
        self.v = 1

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)
        obs, info = self.env.reset(seed=self.seed)
        self.steps = 0
        self.done = True
        return obs, info


    def step(self, action):
        self.last_obs = self.obs
        self.steps = self.steps + 1
        obs, rew, done, truncated, info = self.env.step(action)

        self.obs = obs
        # print(max(self.obs[28:44]))
        goal_dist = 3 - 3 * max(self.obs[12:28])
        # hazard_dist = 3 - 3 * max(self.obs[28:44])
        self.goal_dist = goal_dist
        # self.hazard_dist = hazard_dist
        rho_goal = self.radius * 2 - goal_dist
        self.reward_cache.append(rho_goal)


        if goal_dist <= 0.4:
            done = True


        deadline = self.total_time - self.steps
        # print(deadline)
        if deadline > 0:
            final_reward = max(self.reward_cache)
            final_reward = -final_reward
        else:
            final_reward = max(self.reward_cache[deadline:])


        if truncated:
            done = True
            # final_reward = -5

        obs[2] = deadline / 50
        self.obs = obs


        return obs, rew, done, truncated, info


