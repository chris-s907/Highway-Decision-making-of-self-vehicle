from DecisionMakingEnv import HighwayEnv
from utility import train
from utility import viz

log_dir = "log/"
env = HighwayEnv()

method = 'PPO' # 'DQN', 'A2C', 'PPO', 'RecurrentPPO'
CONTINUE = False # Continue learning
model = train(method, env, 3e5, log_dir, verbose=0, continual=CONTINUE, force_update=0) 
# model = PPO.load(log_dir+"/PPO/best_model.zip", env=env)
viz(model, env, method)