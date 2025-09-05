import numpy as np
import gym
from stable_baselines3 import PPO
import torch
from torch_geometric.nn.conv import GCNConv
from torch_geometric import utils
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import pickle
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix  # Import coo_matrix from scipy.sparse
from torch_geometric.utils import dense_to_sparse


class CustomGNN_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dimension: int = 256, nodevars_dimension=2):
        super().__init__(observation_space, features_dimension)
        self.gcnconv1 = GCNConv(nodevars_dimension, features_dimension)
        self.gcnconv2 = GCNConv(features_dimension, features_dimension)
        self.linear = torch.nn.Linear(features_dimension, features_dimension)





    def forward(self, data):
        statevec = np.stack((data['BusVoltage'], data['BusAngle']), axis=-1)
        statevec = torch.from_numpy(statevec).float()

        adjacency_matrix = np.array(data["Adjacency"])
        print("Original adjacency_matrix shape:", adjacency_matrix.shape)
        print("Original adjacency_matrix size:", adjacency_matrix.size)

        if adjacency_matrix.shape == (1, 118, 118):
            # No need to reshape if the shape is already (1, 118, 118)
            #adjacency_tensor = torch.tensor(adjacency_matrix)
            adjacency_matrix = adjacency_matrix.reshape((118,118))
        elif adjacency_matrix.shape == (2, 118, 118):
            # Reshape to (1, 118, 118)
            adjacency_matrix = adjacency_matrix[0, :, :]
            adjacency_matrix = adjacency_matrix.reshape((118,118))
            #adjacency_tensor = torch.tensor(adjacency_matrix.reshape((1, 118, 118)))

##        adjacency_matrix = np.array(data["Adjacency"])
##        if len(adjacency_matrix.shape) == 2:
##            adjacency_matrix = np.expand_dims(adjacency_matrix, axis=0)
##
##        adjacency_matrix = adjacency_matrix.reshape((118,118))

        adjacency_tensor = torch.tensor(adjacency_matrix)

        
        edge_index = adjacency_tensor.nonzero().t().contiguous()

        
        X = self.gcnconv1(statevec, edge_index)
        X = torch.relu(X)
        X = self.gcnconv2(X, edge_index)
        X = torch.relu(X)
        X = self.linear(X)
        X = torch.relu(X)
        finalFeat = X.mean(dim=1)
        return finalFeat


        
##    def forward(self, data):
##        statevec = np.stack((data['BusVoltage'],
##                                      data['BusAngle']
##                                      ),axis=-1)
##        statevec = torch.from_numpy(statevec).float()
##        adjacency_matrix = np.array(data["Adjacency"])
##
##        # Keep only the first 13924 elements
##        flattened_array = adjacency_matrix[:118, :118].flatten()
##        #flattened_array = adjacency_matrix.flatten()[:13924]
##
##        
##        #adjacency_matrix = adjacency_matrix[:13924]
##        adjacency_matrix = flattened_array.reshape((118,118))
##        adjacency_tensor = torch.tensor(adjacency_matrix)
##        edge_index = adjacency_tensor.nonzero().t().contiguous()
##        
##        X = self.gcnconv1(statevec, edge_index)
##        X = torch.relu(X)
##        X = self.gcnconv2(X, edge_index)
##        X = torch.relu(X)
##        X = self.linear(X)
##        X = torch.relu(X)
##        finalFeat = X.mean(dim=1)
##        return finalFeat 

