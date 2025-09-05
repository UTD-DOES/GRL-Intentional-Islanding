import torch as th
from stable_baselines3.common.policies import BasePolicy
import gym
from gym import spaces
import math
import random
from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.type_aliases import Schedule
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    MlpExtractor
)
from typing import NamedTuple
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
)

from FeatureExtractor import CustomGNN_Extractor



class CustomPolicy(BasePolicy):

    def __init__(
            self,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            lr_schedule: Schedule,
            net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
            activation_fn: Type[th.nn.Module] = th.nn.Tanh,
            ortho_init: bool = True,
            use_sde: bool = False,
            log_std_init: float = 0.0,
            full_std: bool = True,
            sde_net_arch: Optional[List[int]] = None,
            use_expln: bool = False,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
            device: Union[th.device, str] = "cpu"
            ):    
        super(CustomPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output)


        features_dimension = features_extractor_kwargs['features_dimension'] # used to customize our features extractor 
        self.nodevars_dimension = features_extractor_kwargs['nodevars_dimension']

        self.features_extractor = CustomGNN_Extractor(
            nodevars_dimension = self.nodevars_dimension,
            features_dimension = features_dimension,
            observation_space = observation_space
        )
        self.value_net = th.nn.Sequential(th.nn.Linear(features_dimension, features_dimension, bias=True),
                                          th.nn.Linear(features_dimension, 1, bias=True), th.nn.Tanh())

        self.net_arch = net_arch
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde)       
        self.action_net = self.action_dist.proba_distribution_net(latent_dim=features_dimension)


        
        
        self.activation_fn = activation_fn
        
        #self.action_dist = make_proba_distribution(action_space, use_sde=use_sde)

        
        self.mlp_extractor = MlpExtractor( 
            features_dimension,
            net_arch = self.net_arch, # specifies the network architecture of the MLP extractor
            activation_fn = self.activation_fn,
            device = self.device,
        )
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        
        
    def forward(self, obs: th.Tensor, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features) # both the hidden policy and value net has mlp_extractor

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf) # the hidden value net is succeeded by another set of layers        
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        actions = distribution.get_actions(deterministic=True)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob        
    
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi, obs)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()
    
    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        actions, values, log_prob = self.forward(observation, deterministic=deterministic)
        #return th.tensor([actions])
        return actions
    
    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)    
    
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, obs) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

 
        # mean_actions[obs["ActionMasking"].to(th.bool)] += -1000000*(mean_actions[obs["ActionMasking"].to(th.bool)]).abs() #-10000000# -math.inf


        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            # Here mean_actions are the logits before the softmax
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            # Here mean_actions are the flattened logits
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            # Here mean_actions are the logits (before rounding to get the binary actions)
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")


    def make_proba_distribution(
    action_space: gym.spaces.Space, use_sde: bool = False, dist_kwargs: Optional[Dict[str, Any]] = None) -> Distribution:

        if dist_kwargs is None:
            dist_kwargs = {}

        if isinstance(action_space, spaces.Box):
            cls = StateDependentNoiseDistribution if use_sde else DiagGaussianDistribution
            return cls(get_action_dim(action_space), **dist_kwargs)
        elif isinstance(action_space, spaces.Discrete):
            return CategoricalDistribution(action_space.n, **dist_kwargs)
        elif isinstance(action_space, spaces.MultiDiscrete):
            return MultiCategoricalDistribution(action_space.nvec, **dist_kwargs)
        elif isinstance(action_space, spaces.MultiBinary):
            return BernoulliDistribution(action_space.n, **dist_kwargs)
        else:
            raise NotImplementedError(
            "Error: probability distribution, not implemented for action space"
            f"of type {type(action_space)}."
            " Must be of type Gym Spaces: Box, Discrete, MultiDiscrete or MultiBinary."
            )    
