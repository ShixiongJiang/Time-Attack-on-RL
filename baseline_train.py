from stable_baselines3 import PPO
import torch as th
from SA_MDP_env import safetygoal1_env
from fsrl.agent import PPOLagAgent
from fsrl.utils import TensorboardLogger
import torch
env = safetygoal1_env(render_mode=None)

# policy_kwargs = dict(activation_fn=th.nn.ReLU,
#                      net_arch=dict(pi=[128, 64], vf=[128, 64]))
# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log="./ppo_pointgoal0_TimeState/")

# model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)



obs, info = env.reset()

task = "Safetypointgoal-v1"
logger = TensorboardLogger("logs", log_txt=True, name=task)
# init the PPO Lag agent with default parameters
agent = PPOLagAgent(env, logger)
agent.learn(env, env, epoch=50)
torch.save(agent.policy.state_dict(), "model/PPOLag_policy_for_pointgoal1_baseline.pth")

# env.render("human")
# j = 0
# reach = 0
# y = []
# u = []
#
#
#
# total_reward = 0
# total_reach = 0
# total_violate = 0
# eposide = 0
# while eposide < 1:
#
#     # action, _state = model.predict(obs, deterministic=True)
#     # action = env.get_op_action()
#     action = [0, 0]
#     obs, reward, done, trun, info = env.step(action)
#     print(obs[12:28])
#     if done or trun:
#         eposide += 1
#         print(env.steps)
#         # print(done)
#         env.reset()



