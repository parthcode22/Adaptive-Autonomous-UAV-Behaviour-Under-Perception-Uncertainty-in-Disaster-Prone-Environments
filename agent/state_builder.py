"""
state_builder.py — PPO State Vector Assembly
===============================================
SURE-UAV RL Navigation Agent

Combines the 14-dim UPS vector (from fusion_engine.py) with
distance_to_last_signal and the previous action's one-hot encoding
into the final 24-dimensional PPO state vector.

This is the boundary between the perception/fusion layer and the RL
agent — UPS and FusionEngine have no knowledge of actions or RL
concepts; this file is where those worlds connect.
"""

from __future__ import annotations
import numpy as np

from fusion_1.ups_vector import UnifiedPerceptualState
from fusion_1.fusion_engine import FusionEngine
from .action_space import UAVAction, action_to_onehot, num_actions

def build_ppo_state(
    ups          : UnifiedPerceptualState,
    fusion_engine : FusionEngine,
    prev_action    : UAVAction,
) -> np.ndarray:
    """
    Assemble the full 24-dimensional PPO state vector.

    Parameters
    ----------
    ups           : UnifiedPerceptualState from the most recent fuse() call
    fusion_engine : the FusionEngine instance that produced ups (for
                    accessing get_distance_to_last_signal())
    prev_action   : the UAVAction taken in the previous timestep

    Returns
    -------
    np.ndarray of shape (24,), dtype float32

    Layout
    ------
    [0:14]  — UPS's own 14-dim vector (scene_confidence ... yaw_rate)
    [14]    — distance_to_last_signal, normalized [0, 1]
    [15:24] — prev_action one-hot, 9 dims
    """
    base_14 = ups.to_ppo_state()                                  # shape (14,)
    distance = np.array(
        [fusion_engine.get_distance_to_last_signal()], dtype=np.float32
    )                                                              # shape (1,)
    prev_action_vec = np.array(
        action_to_onehot(prev_action), dtype=np.float32
    )                                                              # shape (9,)

    full_state = np.concatenate([base_14, distance, prev_action_vec])
    return full_state.astype(np.float32)
STATE_DIM = 14 + 1 + num_actions()   # 14 + 1 + 9 = 24


if __name__ == "__main__":
    from fusion_1.fusion_engine import GaborPipelineOutput, LidarPipelineOutput, OdometryInput

    print("=" * 70)
    print("state_builder.py — Self-Test")
    print(f"Expected STATE_DIM = {STATE_DIM}")
    print("=" * 70)

    engine = FusionEngine()
    gabor  = GaborPipelineOutput(global_confidence=0.8, informativeness=0.75, tier='SAFE')
    lidar  = LidarPipelineOutput(p_surface=0.7, p_smoke=0.1, p_unknown=0.2,
                                  spatial_consistency=0.8, valid_echo_fraction=0.9)
    odom   = OdometryInput(vx=1.0, x=0.0, y=0.0, z=0.0)

    ups = engine.fuse(gabor, lidar, odom, radar_flag=0)
    state = build_ppo_state(ups, engine, prev_action=UAVAction.FORWARD_NORMAL)

    print(f"\nState shape : {state.shape}")
    print(f"State dtype : {state.dtype}")
    print(f"Full vector : {np.round(state, 3)}")
    print(f"\nBreakdown:")
    print(f"  [0:14]  UPS fields            : {np.round(state[0:14], 3)}")
    print(f"  [14]    distance_to_signal    : {round(float(state[14]), 3)}")
    print(f"  [15:24] prev_action one-hot   : {state[15:24]}")

    assert state.shape == (STATE_DIM,), "STATE_DIM MISMATCH"
    print("\nSelf-test PASSED.")

