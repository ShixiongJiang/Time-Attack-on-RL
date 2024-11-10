import gymnasium
import safety_gymnasium

import numpy as np
import math
from copy import deepcopy
from stable_baselines3 import A2C, PPO

class SafetyPointGoal1_delay(gymnasium.Env):
    def __init__(self, config=None, seed=None, render_mode='human', semantics_method_tradition=False):
        # super(SafetyPointGoal1, self).__init__()
        self.total_time = 100
        self.safe_dis = 0.4
        self.obs = None
        self.hazard_dist = None
        self.goal_dist = None
        env_id = 'SafetyPointGoal0-v0'
        safety_gymnasium_env = safety_gymnasium.make(env_id, render_mode=render_mode)
        self.env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
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
        self.done = False
        self.reward_cache = []
        return obs, info

    def get_op_action(self):
        obstacle = np.argmax(self.obs[28:44])

        turn = self.obs[4]
        velo = self.obs[3]
        if obstacle > 4 and obstacle < 12:
            action0 = -1
        else:
            action0 = 1
        if obstacle > 4 and obstacle < 12:
            if obstacle < 8:
                action1 = -1 * (8 - obstacle) / 3 * abs(velo * 5) / (3 - 3 * max(self[28:44]))
            elif obstacle > 8:
                action1 = 1 * (obstacle - 8) / 3 * abs(velo * 5) / (3 - 3 * max(self[28:44]))
            else:
                action1 = 0
        else:
            if obstacle <= 4:
                action1 = 1 * (obstacle + 1) / 3 * abs(velo * 5) / (3 - 3 * max(self[28:44]))
            elif obstacle >= 13 and obstacle < 16:
                action1 = -1 * (15 - obstacle) / 3 * abs(velo * 5) / (3 - 3 * max(self[28:44]))
            else:
                action1 = 0
        action = [action0, action1]
        return action


    def step(self, action):

        self.last_obs = self.obs
        self.steps = self.steps + 1
        action[0] = action[0] / 20
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

        # print(final_reward)

        if truncated:
            done = True
            # final_reward = -5

        obs[2] = deadline / 50
        self.obs = obs
        # print(done)
        print(info)
        return obs, final_reward, done, truncated, info

    def trandition_semantics(self, goal_dist, hazard_dist):
        rho_goal = self.radius * 2 - goal_dist
        rho_safe = (hazard_dist - self.safe_dis) * 10

        self.reward_cache.append(rho_goal)
        self.avoid_reward_cache.append(rho_safe)
        if self.steps < 10:
            rho_goal_final = max(self.reward_cache)
        else:
            rho_goal_final = max(self.reward_cache[-10:])
        if self.steps < 10:
            rho_safe_always = min(self.avoid_reward_cache)
        else:
            rho_safe_always = min(self.avoid_reward_cache[-10:])

        final_reward = min(rho_goal_final, rho_safe_always)

        self.final_reward_cache.append(final_reward)
        return final_reward

    def softmax_semantics(self, goal_dist, hazard_dist):
        rho_goal = self.radius * 2 - goal_dist
        rho_safe = (hazard_dist - self.safe_dis) * 10

        self.reward_cache.append(rho_goal)
        self.avoid_reward_cache.append(rho_safe)
        if self.steps < 10:
            rho_goal_final = max(self.reward_cache)
        else:
            rho_goal_final = max(self.reward_cache[-10:])
        if self.steps < 10:
            rho_safe_always = min(self.avoid_reward_cache)
        else:
            rho_safe_always = min(self.avoid_reward_cache[-10:])
        rho_min = min(rho_goal_final, rho_safe_always)
        if rho_min == 0:
            final_reward = 0
            return final_reward
        rho_goal_tilde = (rho_goal_final - rho_min) / rho_min
        rho_safe_tilde = (rho_safe_always - rho_min) / rho_min
        if rho_min < 0:
            final_reward = rho_min * math. exp(rho_goal_tilde) * math. exp(rho_goal_tilde * self.v)
            final_reward = final_reward + rho_min * math.exp(rho_safe_tilde) * math.exp(rho_safe_tilde * self.v)
            final_reward = final_reward / (math. exp(rho_goal_tilde * self.v) +  math.exp(rho_safe_tilde * self.v))

        if rho_min > 0:
            final_reward = rho_goal_final * math. exp(-rho_goal_tilde * self.v)
            final_reward = final_reward + rho_safe_always * math. exp(-rho_safe_tilde * self.v)
            final_reward = final_reward / (math. exp(-rho_goal_tilde * self.v) +  math.exp(-rho_safe_tilde * self.v))
        return final_reward

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


