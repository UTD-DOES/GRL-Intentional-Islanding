import gymnasium as gym
from pypsse_Gym.envs.pypsseEnv118 import pypsseEnv118

gym.envs.registration.register(id='pypsseEnv118-v0', entry_point='pypsse_Gym.envs:pypsseEnv118')


