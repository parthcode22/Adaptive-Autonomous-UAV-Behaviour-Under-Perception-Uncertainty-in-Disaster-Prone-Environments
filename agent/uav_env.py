"""
uav_env.py — Mock SAR Training Environment
=============================================
SURE-UAV RL Navigation Agent

A Gymnasium-compatible environment for training the PPO agent BEFORE
Gazebo/PX4 simulation exists. Drone position is simulated directly from
chosen action kinematics; Gabor/LiDAR sensor outputs are synthesized
based on the drone's position relative to a fixed hazard zone, fed
through the REAL FusionEngine so navigation_state/UPS math is genuine,
not faked.

Fixed mission layout (v1 — randomization to be added later):
    Point A (start) : (0, 0, 0)
    Point B (goal)  : (80, 0, 0)
    Hazard zone     : center (40, 0, 0), radius 15m
    Victim location : (45, 5, 0), detection radius 5m
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from fusion_1.fusion_engine import (
    FusionEngine, GaborPipelineOutput, LidarPipelineOutput, OdometryInput,
)
from .action_space import UAVAction, get_kinematics, num_actions, index_to_action
from .state_builder import build_ppo_state, STATE_DIM
from .reward_function import RewardContext, RewardConstants, compute_reward


class MissionConstants:
    """Fixed mock mission layout — v1, no randomization yet."""
    POINT_A          = np.array([0.0, 0.0, 0.0])
    POINT_B          = np.array([80.0, 0.0, 0.0])
    HAZARD_CENTER    = np.array([40.0, 0.0, 0.0])
    HAZARD_RADIUS    = 15.0
    VICTIM_POSITION  = np.array([45.0, 5.0, 0.0])
    VICTIM_RADIUS    = 5.0
    MAX_STEPS        = 500
    TIMESTEP_DT      = 0.5   # seconds per env step


class UAVNavigationEnv(gym.Env):
    """
    Mock SAR navigation environment. See module docstring for the
    fixed mission layout this version trains against.
    """

    def __init__(self, consts: MissionConstants = MissionConstants()):
        super().__init__()
        self.consts = consts

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(num_actions())

        self.fusion_engine = FusionEngine(max_signal_distance=50.0)

        # Episode state — initialized properly in reset()
        self.position                = self.consts.POINT_A.copy()
        self.yaw                      = 0.0
        self.step_count                 = 0
        self.prev_action                  = UAVAction.HOVER_OBSERVE
        self.prev_distance_to_goal           = 1.0
        self.prev_distance_to_signal           = 1.0
        self.signal_ever_found                   = False
        self.last_ups                               = None

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.position                = self.consts.POINT_A.copy()
        self.yaw                      = 0.0
        self.step_count                 = 0
        self.prev_action                  = UAVAction.HOVER_OBSERVE
        self.signal_ever_found               = False

        self.fusion_engine.reset()

        self.prev_distance_to_goal   = self._normalized_distance_to_goal()
        self.prev_distance_to_signal  = 1.0

        gabor, lidar, radar_flag = self._synthesize_sensors()
        odom = OdometryInput(
            vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0,
            x=self.position[0], y=self.position[1], z=self.position[2],
        )
        ups = self.fusion_engine.fuse(gabor, lidar, odom, radar_flag=radar_flag)
        self.last_ups = ups

        state = build_ppo_state(ups, self.fusion_engine, self.prev_action)
        info = {}
        return state, info

    # ------------------------------------------------------------------
    def step(self, action_index: int):
        action = index_to_action(int(action_index))
        kin = get_kinematics(action)

        dt = self.consts.TIMESTEP_DT
        dx = kin.forward_velocity * np.cos(self.yaw) * dt
        dy = kin.forward_velocity * np.sin(self.yaw) * dt
        dz = kin.vertical_velocity * dt

        self.position = self.position + np.array([dx, dy, dz])
        self.yaw      = self.yaw + kin.yaw_rate * dt

        distance_moved = float(np.linalg.norm([dx, dy, dz]))

        gabor, lidar, radar_flag = self._synthesize_sensors()
        odom = OdometryInput(
            vx=kin.forward_velocity, vy=0.0, vz=kin.vertical_velocity,
            yaw_rate=kin.yaw_rate,
            x=self.position[0], y=self.position[1], z=self.position[2],
        )
        ups = self.fusion_engine.fuse(gabor, lidar, odom, radar_flag=radar_flag)
        self.last_ups = ups

        distance_to_goal = self._normalized_distance_to_goal()
        distance_to_signal = self.fusion_engine.get_distance_to_last_signal()

        radar_just_triggered = (radar_flag == 1)
        if radar_just_triggered:
            self.signal_ever_found = True

        goal_reached = distance_to_goal * RewardConstants.MAX_MISSION_DISTANCE \
                       <= RewardConstants.SUCCESS_RADIUS

        self.step_count += 1
        is_timeout = self.step_count >= self.consts.MAX_STEPS

        ctx = RewardContext(
            navigation_state=ups.navigation_state,
            obstacle_confidence=ups.obstacle_confidence,
            action=action,
            prev_action=self.prev_action,
            distance_to_goal=distance_to_goal,
            prev_distance_to_goal=self.prev_distance_to_goal,
            distance_to_last_signal=distance_to_signal,
            prev_distance_to_signal=self.prev_distance_to_signal,
            signal_ever_found=self.signal_ever_found,
            radar_just_triggered=radar_just_triggered,
            distance_moved=distance_moved,
            goal_reached=goal_reached,
            is_timeout=is_timeout,
        )
        result = compute_reward(ctx)

        self.prev_distance_to_goal    = distance_to_goal
        self.prev_distance_to_signal   = distance_to_signal
        self.prev_action                  = action

        state = build_ppo_state(ups, self.fusion_engine, self.prev_action)

        terminated = result.crashed or goal_reached
        truncated  = is_timeout and not terminated

        info = {
            "reward_breakdown": result,
            "position": self.position.copy(),
            "navigation_state": ups.navigation_state.value,
        }

        return state, result.total, terminated, truncated, info

    # ------------------------------------------------------------------
    def _normalized_distance_to_goal(self) -> float:
        raw_distance = float(np.linalg.norm(self.position - self.consts.POINT_B))
        return float(np.clip(raw_distance / RewardConstants.MAX_MISSION_DISTANCE, 0.0, 1.0))

    def _hazard_proximity(self) -> float:
        """
        0.0 = far from hazard zone (clear), 1.0 = at hazard center (worst).
        Linearly interpolates within HAZARD_RADIUS, zero outside it.
        """
        dist_to_hazard = float(np.linalg.norm(self.position - self.consts.HAZARD_CENTER))
        if dist_to_hazard >= self.consts.HAZARD_RADIUS:
            return 0.0
        return 1.0 - (dist_to_hazard / self.consts.HAZARD_RADIUS)

    def _synthesize_sensors(self):
        """
        Position-driven synthetic Gabor/LiDAR outputs. Inside the hazard
        zone, confidence degrades proportionally to proximity to its
        center — this drives the REAL FusionEngine into CAUTION/DANGER/
        BLIND naturally, rather than faking navigation_state directly.
        """
        proximity = self._hazard_proximity()

        global_conf = float(np.clip(0.90 - 0.85 * proximity, 0.05, 0.90))
        info        = global_conf * 0.9
        tier        = "SAFE" if global_conf >= 0.45 else "DANGER"

        gabor = GaborPipelineOutput(
            global_confidence=global_conf,
            informativeness=info,
            tier=tier,
            scale_coherence_small=global_conf,
            scale_coherence_normal=global_conf * 0.9,
            scale_coherence_large=global_conf * 0.8,
            regional_left=global_conf, regional_center=global_conf, regional_right=global_conf,
            danger_pixel_fraction=proximity,
        )

        lidar = LidarPipelineOutput(
            p_surface=float(np.clip(0.85 - 0.80 * proximity, 0.05, 0.85)),
            p_smoke=float(np.clip(0.05 + 0.80 * proximity, 0.05, 0.85)),
            p_unknown=0.10,
            spatial_consistency=float(np.clip(0.85 - 0.70 * proximity, 0.10, 0.85)),
            valid_echo_fraction=float(np.clip(0.90 - 0.60 * proximity, 0.20, 0.90)),
            beta_estimate=float(proximity * 1.2),
            obstacle_proximity=proximity,
        )

        dist_to_victim = float(np.linalg.norm(self.position - self.consts.VICTIM_POSITION))
        radar_flag = 1 if dist_to_victim <= self.consts.VICTIM_RADIUS else 0

        return gabor, lidar, radar_flag


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 78)
    print("uav_env.py — Self-Test")
    print("=" * 78)

    env = UAVNavigationEnv()
    state, info = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    assert state.shape == (STATE_DIM,)

    print(f"\nRunning a hardcoded policy: FORWARD_NORMAL every step")
    print(f"{'step':>5} {'pos_x':>8} {'nav_state':>10} {'reward':>8} {'done':>6}")

    total_reward = 0.0
    for i in range(200):
        action = UAVAction.FORWARD_NORMAL.value
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 20 == 0 or terminated or truncated:
            print(f"{i:>5} {info['position'][0]:>8.2f} {info['navigation_state']:>10} "
                  f"{reward:>8.3f} {terminated or truncated}")

        if terminated or truncated:
            print(f"\nEpisode ended at step {i}. Terminated={terminated} Truncated={truncated}")
            break

    print(f"\nTotal reward accumulated: {total_reward:.3f}")
    print("\n" + "=" * 78)
    print("Self-test complete.")
    print("=" * 78)