import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from tqdm import tqdm
from TrainingConfiguration import get_config
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from pypsseEnv118 import pypsseEnv118
from CustomizedPolicy import CustomPolicy
from typing import Callable  # Import Callable
from FeatureExtractor import CustomGNN_Extractor

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.mean_rewards = []
        self.num_steps = []
        self.num_actions = []
        self.best_mean_reward = -np.inf  # Initialize with a very low value

    def _on_step(self) -> bool:
        self.num_steps.append(self.num_timesteps)
        self.num_actions.append(self.model.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        episodes_rewards = self.locals["ep_infos"]
        mean_reward = np.mean([ep_info["r"] for ep_info in episodes_rewards])
        self.mean_rewards.append(mean_reward)
        tqdm.write(f"Mean Reward: {mean_reward}, Num Steps: {self.num_steps[-1]}, Num Actions: {self.num_actions[-1]}")

        # Check if the current mean reward is the best so far
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            # Save the best model
            self.model.save("best_model")

def learning_rate_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return initial_value
    return func

def make_env(rank, seed=0):
    def _init():
        env = pypsseEnv118()
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    # Load configuration
    training_config = get_config()
    #tb_logger_location = training_configuratn.Logger

    # Create environment
    env = pypsseEnv118()

    # Define policy kwargs
    policy_kwargs = dict(
        features_extractor_class=CustomGNN_Extractor,
        features_extractor_kwargs=dict(features_dimension=training_config.FeatureDim, nodevars_dimension=2),
        net_arch=[
            dict(
                pi=[training_config.FeatureDim, training_config.FeatureDim],
                vf=[training_config.FeatureDim, training_config.FeatureDim]
            )],
        device=torch.device("cpu")  # Change this if CUDA is available
    )

    # Create the PPO model
    model = PPO(
        policy=CustomPolicy,
        env=env,
        tensorboard_log=training_config.Logger,
        #tensorboard_log=tb_logger_location,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=training_config.TotalSteps,
        batch_size=training_config.BatchSize,
        gamma=training_config.Gamma,
        learning_rate=learning_rate_schedule(training_config.LearningRate),
        ent_coef=training_config.EntropyCoef
    )

    # Create callbacks
    callback = CustomCallback()
    checkpoint_callback = CheckpointCallback(save_freq=training_config.SaveFreq, save_path=training_config.ModelSave)

    # Train the model
    with tqdm(total=training_config.TotalSteps, desc="Training") as pbar:
        model.learn(
            total_timesteps=training_config.TotalSteps,
            callback=callback,
            reset_num_timesteps=False,
            tb_log_name='ppo',
            log_interval=10,
            #callback_on_new_best=checkpoint_callback
        )
        pbar.update(model.num_timesteps)
