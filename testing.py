import numpy as np
import gym

import torch as th
import torch.nn as nn

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from pypsseEnv118 import pypsseEnv118
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from training import*


num_actions=21
env = pypsseEnv118()

observations = env.test_model()


log_dir = "G"
model = PPO.load(log_dir + "_RLGCN", env=env)



observations['BusVoltage'] = np.reshape(observations['BusVoltage'], (1,observations['BusVoltage'].shape[0]))
observations['BusAngle'] = np.reshape(observations['BusAngle'], (1,observations['BusAngle'].shape[0]))

action, _states = model.predict(observations, deterministic=True)


action = np.reshape(action,(num_actions,))
newobservations, rewards, dones, info = env.step(action)

