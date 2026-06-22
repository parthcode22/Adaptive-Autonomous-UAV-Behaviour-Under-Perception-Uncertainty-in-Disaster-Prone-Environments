"""
ppo_network.py — PPO Policy/Value Network Architecture
==========================================================
SURE-UAV RL Navigation Agent

Defines a custom feature extractor for Stable-Baselines3's PPO, given
our 24-dimensional state vector (from state_builder.py) and 9 discrete
actions (from action_space.py).

Architecture: shared trunk (Dense 128 -> ReLU -> Dense 128 -> ReLU),
with SB3's PPO automatically attaching separate policy and value heads
on top of this shared feature extractor.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .action_space import num_actions
from .state_builder import STATE_DIM


class SureUAVFeatureExtractor(BaseFeaturesExtractor):
    """
    Shared trunk feature extractor for the SURE-UAV PPO agent.

    Takes the 24-dim state vector and produces a 128-dim feature
    representation. SB3's PPO then attaches:
        - a policy head: Linear(128 -> 9) + Softmax (action probabilities)
        - a value head:  Linear(128 -> 1)            (state value estimate)
    on top of this extractor automatically — we do not define those
    heads ourselves.

    Parameters
    ----------
    observation_space : gymnasium.spaces.Box, shape (24,)
    features_dim       : output dimension of this extractor (default 128)
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]
        assert input_dim == STATE_DIM, (
            f"Expected state dim {STATE_DIM}, got {input_dim}. "
            f"Check state_builder.py and the environment's observation_space."
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


def get_policy_kwargs(features_dim: int = 128) -> dict:
    """
    Returns the policy_kwargs dict to pass into SB3's PPO constructor,
    wiring in our custom SureUAVFeatureExtractor.

    Usage (in train_ppo.py):
        from stable_baselines3 import PPO
        model = PPO("MlpPolicy", env, policy_kwargs=get_policy_kwargs(), ...)
    """
    return dict(
        features_extractor_class=SureUAVFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=features_dim),
        net_arch=[],   # empty: policy/value heads attach directly to our
                       # 128-dim extractor output, no extra MLP layers added
    )