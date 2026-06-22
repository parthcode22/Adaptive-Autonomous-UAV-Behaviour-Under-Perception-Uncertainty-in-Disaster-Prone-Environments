"""
action_space.py — Discrete Action Space Definition
=====================================================
SURE-UAV RL Navigation Agent

Defines the 9 discrete actions the PPO agent can choose from, along with
the kinematic parameters each action maps to, per the locked SURE-UAV
architecture (Section II: Sensor & Input Layer, kinematic speeds).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict

class UAVAction(Enum):
    """
    The 9 discrete actions available to the PPO policy.
    Integer values double as array indices for one-hot encoding.
    """
    FORWARD_NORMAL  = 0
    FORWARD_SLOW    = 1
    TURN_LEFT       = 2
    TURN_RIGHT      = 3
    HOVER_OBSERVE   = 4
    YAW_SCAN_360    = 5
    BACKTRACK       = 6
    ASCEND          = 7
    DESCEND         = 8


@dataclass

class ActionKinematics:
    forward_velocity: float  # m/s
    yaw_rate: float          # rad/s
    vertical_velocity: float  # m/s

ACTION_KINEMATICS: Dict[UAVAction, ActionKinematics] = {
    UAVAction.FORWARD_NORMAL : ActionKinematics(forward_velocity= 1.75, yaw_rate= 0.0,  vertical_velocity=0.0),
    UAVAction.FORWARD_SLOW   : ActionKinematics(forward_velocity= 0.70, yaw_rate= 0.0,  vertical_velocity=0.0),
    UAVAction.TURN_LEFT      : ActionKinematics(forward_velocity= 0.70, yaw_rate= 0.5,  vertical_velocity=0.0),
    UAVAction.TURN_RIGHT     : ActionKinematics(forward_velocity= 0.70, yaw_rate=-0.5,  vertical_velocity=0.0),
    UAVAction.HOVER_OBSERVE  : ActionKinematics(forward_velocity= 0.0,  yaw_rate= 0.0,  vertical_velocity=0.0),
    UAVAction.YAW_SCAN_360   : ActionKinematics(forward_velocity= 0.0,  yaw_rate= 0.8,  vertical_velocity=0.0),
    UAVAction.BACKTRACK      : ActionKinematics(forward_velocity=-1.00, yaw_rate= 0.0,  vertical_velocity=0.0),
    UAVAction.ASCEND         : ActionKinematics(forward_velocity= 0.0,  yaw_rate= 0.0,  vertical_velocity= 0.5),
    UAVAction.DESCEND        : ActionKinematics(forward_velocity= 0.0,  yaw_rate= 0.0,  vertical_velocity=-0.5),
}

def get_kinematics(action: UAVAction) -> ActionKinematics:
    """Look up the kinematic parameters for a given action."""
    return ACTION_KINEMATICS[action]


def action_to_index(action: UAVAction) -> int:
    """Convert a UAVAction to its integer index (matches Enum value)."""
    return action.value


def index_to_action(index: int) -> UAVAction:
    """Convert an integer index back to a UAVAction."""
    return UAVAction(index)


def num_actions() -> int:
    """Total number of discrete actions — used for network output layer size."""
    return len(UAVAction)


def action_to_onehot(action: UAVAction) -> list[float]:
    """
    Convert a UAVAction into a one-hot encoded list, for the
    prev_action_onehot field in the PPO state vector.
    """
    vec = [0.0] * num_actions()
    vec[action.value] = 1.0
    return vec 