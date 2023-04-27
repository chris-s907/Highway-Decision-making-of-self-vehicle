import numpy as np
from DecisionMakingEnv import HighwayEnv as HighwayEnv_rl
from EgoDecisionTree import HighwayEnv as HighwayEnv_dt
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN, A2C, PPO

env = HighwayEnv_rl()
env.reset()
timestep = 5000

collision_time     = 0
outOfBoundary_time = 0
lane_change_time   = 0
v_list             = []
a_list             = []
num_scenario       = 0

log_dir = "log/"

method = 'PPO' # 'DQN', 'A2C', 'PPO', 'RecurrentPPO', 'decisionTree'

w_v  = 0.001
w_a  = 0.001
w_j  = 0.002
w_c  = 120
w_lc = 15


if method == 'DQN':
    model = DQN.load(log_dir+"/DQN/best_model.zip", env=env)
    
elif method == 'A2C':
    model = A2C.load(log_dir+"/A2C/best_model.zip", env=env)
    
elif method == 'PPO':
    model = PPO.load(log_dir+"/PPO/best_model.zip", env=env)

elif method == 'RecurrentPPO':
    model = RecurrentPPO.load(log_dir+"/RecurrentPPO/best_model.zip", env=env)

if method in ["DQN", "PPO", "A2C"]:
    vec_env = model.get_env()
    obs = vec_env.reset() 
    for i in range(timestep):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        # vec_env.render()
        # VecEnv resets automatically
        if done:
            collision_time     += env.collision_time
            outOfBoundary_time += env.outOfBoundary_time
            lane_change_time   += env.lane_change_time
            v_list             += env.v_list
            a_list             += env.a_list
            num_scenario       += env.num_scenario
            obs = vec_env.reset()

elif method == "RecurrentPPO":
    obs = env.reset()
    # cell and hidden state of the LSTM
    lstm_states = None
    num_envs = 1
    # Episode start signals are used to reset the lstm states
    episode_starts = np.ones((num_envs,), dtype=bool)

    for i in range(timestep):
        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        episode_starts = np.ones((num_envs,), dtype=bool) if dones else np.zeros((num_envs,), dtype=bool)
        # env.render()
        # Check if episode has started in any environment and reset environment state
        if np.any(episode_starts):
            collision_time     += env.collision_time
            outOfBoundary_time += env.outOfBoundary_time
            lane_change_time   += env.lane_change_time
            v_list             += env.v_list
            a_list             += env.a_list
            num_scenario       += env.num_scenario
            obs = env.reset()   # Reset observations
            lstm_states = None  # Reset LSTM states
 
elif method == 'decisionTree':
    env = HighwayEnv_dt()
    for i in range(timestep):
        _, reward, done, _ = env.step('dummy')
        # env.render()
 
        if done:
            collision_time     += env.collision_time
            outOfBoundary_time += env.outOfBoundary_time
            lane_change_time   += env.lane_change_time
            v_list             += env.v_list
            a_list             += env.a_list
            num_scenario       += env.num_scenario
            env.reset()
    
    # plt.show(block=False)

collision_time     = env.collision_time
outOfBoundary_time = env.outOfBoundary_time
lane_change_time   = env.lane_change_time
v_list             = env.v_list
a_list             = env.a_list
num_scenario       = env.num_scenario


j_v  = w_v  * sum([(env.ego.target_speed - v)**2 for v in v_list])
j_a  = w_a  * sum([a**2 for a in a_list])
j_j  = w_j  * sum([(a_list[i+1] - a_list[i])**2 for i in range(len(a_list) - 1)])
j_c  = w_c  * collision_time
j_lc = w_lc * lane_change_time
j_total = j_v + j_a + j_j + j_c + j_lc

print(f"j_v     = {j_v}")
print(f"j_a     = {j_a}")
print(f"j_j     = {j_j}")
print(f"j_c     = {j_c}")
print(f"j_lc    = {j_lc}")
print(f"j_total = {j_total}")


print(f"Out-of-boundary rate: {outOfBoundary_time/env.num_scenario:.2f}")
print(f"Collision rate: {collision_time/env.num_scenario:.2f}")
print(f"Lane-changing rate: {lane_change_time/env.num_scenario:.2f}")
print(f"Number of scenarios: {num_scenario}")