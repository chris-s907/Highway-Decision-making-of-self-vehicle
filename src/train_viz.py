import argparse
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, DQN, A2C

from DecisionMakingEnv import HighwayEnv
from utility import train
from utility import viz

parser = argparse.ArgumentParser()

# Define the 'task' argument
parser.add_argument('--viz', choices=['true', 'false'], default='true', help='Whether to visualize the trained model or start training')
parser.add_argument('--method', choices=['DQN', 'A2C', 'PPO', 'RecurrentPPO'], default='dqn', help='Which method to be used for training')
parser.add_argument('--continual', choices=['true', 'false'], default='false', help='Whether to continue learning')
args = parser.parse_args()

log_dir = "log/"
env = HighwayEnv()

if args.viz == 'true':
    if args.method == 'DQN':
        model = DQN.load(log_dir+"DQN/best_model.zip", env=env)
        viz(model, env, 'DQN')
    if args.method == 'A2C':
        model = A2C.load(log_dir+"A2C/best_model.zip", env=env)
        viz(model, env, 'A2C')
    if args.method == 'PPO':
        model = PPO.load(log_dir+"PPO/best_model.zip", env=env)
        viz(model, env, 'PPO')
    if args.method == 'RecurrentPPO':  
        model = RecurrentPPO.load(log_dir+"RecurrentPPO/best_model.zip", env=env)
        viz(model, env, 'RecurrentPPO')
else:
    method = args.method
    CONTINUE = True if args.continual == 'true' else False
    model = train(method, env, 5e5, log_dir, verbose=0, continual=CONTINUE, force_update=0) 