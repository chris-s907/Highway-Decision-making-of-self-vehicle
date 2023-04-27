import os
import numpy as np

from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    :param best_mean_reward: Initial value for best mean reward
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1, best_mean_reward: float = -np.inf):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = best_mean_reward

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward

                    # Save the best mean reward to a file
                    np.save(os.path.join(self.save_path, "best_mean_reward.npy"), self.best_mean_reward)

                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True



def train(method, env, timesteps, log_dir, verbose, continual, force_update):
    policy_kwargs_DQN = {"net_arch" : [128, 256, 256, 128]}
    policy_kwargs_PPO = {"net_arch" : [128, 256, 256, 128]}
    policy_kwargs_A2C = {"net_arch" : [128, 256, 256, 128]}
    if method == 'DQN':
        log_dir += method
        env = Monitor(env, log_dir)
        timesteps = timesteps
        if continual:
            if force_update:
                best_mean_reward = -np.inf
            else:
                best_mean_reward = np.load(log_dir + "/best_model/best_mean_reward.npy")  # Load the best mean reward from the saved model
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, best_mean_reward=best_mean_reward)
            model = DQN.load(log_dir+"/best_model.zip", env=env)
            # model.set_env(env)
            model.learn(total_timesteps = timesteps, callback=callback, reset_num_timesteps=False, tb_log_name=log_dir)
        else:
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
            model = DQN("MlpPolicy", env, batch_size=2048, 
                        buffer_size=50000, 
                        learning_rate=0.0005,
                        target_update_interval=250, 
                        policy_kwargs=policy_kwargs_DQN, 
                        verbose=verbose, 
                        tensorboard_log=log_dir)
            model.learn(total_timesteps=timesteps, callback=callback)
    elif method == 'A2C':
        log_dir += method
        env = Monitor(env, log_dir)
        timesteps = timesteps
        if continual:
            if force_update:
                best_mean_reward = -np.inf
            else:
                best_mean_reward = np.load(log_dir + "/best_model/best_mean_reward.npy")  # Load the best mean reward from the saved model
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, best_mean_reward=best_mean_reward)
            model = A2C.load(log_dir+"/best_model.zip", env=env)
            # model.set_env(env)
            model.learn(total_timesteps = timesteps, callback=callback, reset_num_timesteps=False)
        else:
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
            model = A2C("MlpPolicy", env,
                        learning_rate=0.0005,
                        policy_kwargs=policy_kwargs_A2C, 
                        verbose=verbose, 
                        tensorboard_log=log_dir)
            model.learn(total_timesteps=timesteps, callback=callback)
    elif method == 'PPO':
        log_dir += method
        env = Monitor(env, log_dir)
        timesteps = timesteps
        if continual:
            if force_update:
                best_mean_reward = -np.inf
            else:
                best_mean_reward = np.load(log_dir + "/best_model/best_mean_reward.npy")  # Load the best mean reward from the saved model
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, best_mean_reward=best_mean_reward)
            model = PPO.load(log_dir+"/best_model.zip", env=env)
            # model.set_env(env)
            model.learn(total_timesteps = timesteps, callback=callback, reset_num_timesteps=False)
        else:
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
            model = PPO("MlpPolicy", env, batch_size=2048, 
                        learning_rate=0.0005,
                        policy_kwargs=policy_kwargs_PPO, 
                        verbose=0, 
                        tensorboard_log=log_dir)
            model.learn(total_timesteps=timesteps, callback=callback)
    elif method == 'RecurrentPPO':
        log_dir += method
        env = Monitor(env, log_dir)
        timesteps = timesteps
        if continual:
            if force_update:
                best_mean_reward = -np.inf
            else:
                best_mean_reward = np.load(log_dir + "/best_model/best_mean_reward.npy")  # Load the best mean reward from the saved model
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, best_mean_reward=best_mean_reward)
            model = RecurrentPPO.load(log_dir+"/best_model.zip", env=env, tensorboard_log=log_dir)
            # model.set_env(env)
            model.learn(total_timesteps = timesteps, callback=callback, reset_num_timesteps=False)
        else:
            callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
            model = RecurrentPPO("MlpLstmPolicy", env, batch_size=2048, 
                        learning_rate=0.0005,
                        verbose=0,
                        tensorboard_log=log_dir)
            model.learn(total_timesteps=timesteps, callback=callback)
    else:
        AssertionError("Invalid method")

    return model

def viz(model, env, method):
    if method == "RecurrentPPO":
        obs = env.reset()
        # cell and hidden state of the LSTM
        lstm_states = None
        num_envs = 1
        # Episode start signals are used to reset the lstm states
        episode_starts = np.ones((num_envs,), dtype=bool)

        while True:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            episode_starts = np.ones((num_envs,), dtype=bool) if dones else np.zeros((num_envs,), dtype=bool)
            env.render()
            # Check if episode has started in any environment and reset environment state
            if np.any(episode_starts):
                obs = env.reset()   # Reset observations
                lstm_states = None  # Reset LSTM states

    elif method in ["DQN", "PPO", "A2C"]:

        vec_env = model.get_env()
        obs = vec_env.reset()
        while True:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render()
            # VecEnv resets automatically
            if done:
              obs = vec_env.reset()

    else:
        print("Invalid argument")
