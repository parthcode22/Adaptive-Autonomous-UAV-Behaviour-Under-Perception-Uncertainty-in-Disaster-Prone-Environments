"""
fusion_engine.py — Core Weighted Fusion Engine
===============================================
SURE-UAV Fusion Layer

Combines Gabor + LiDAR pipeline outputs using dynamic sensor weights
to produce the Unified Perceptual State (UPS) vector each cycle.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import time

from .ups_vector import (
    UnifiedPerceptualState,
    NavigationState,
    DominantSensor,
    FusionThresholds,
    make_blind_ups,
)
from .sensor_weights import (
    SensorWeightComputer,
    VisualReliabilityInput,
    LidarReliabilityInput,
)


@dataclass
class GaborPipelineOutput:
    """
    Everything your existing Gabor pipeline produces per frame.
    Fields map directly to CoherenceMapComputer output.
    """
    global_confidence      : float
    informativeness        : float
    tier                   : str     # 'SAFE' or 'DANGER'
    scale_coherence_small  : float = 0.5
    scale_coherence_normal : float = 0.5
    scale_coherence_large  : float = 0.5
    regional_left          : float = 0.5
    regional_center        : float = 0.5
    regional_right         : float = 0.5
    danger_pixel_fraction  : float = 0.0

    def __post_init__(self):
        self.global_confidence      = float(np.clip(self.global_confidence,      0.0, 1.0))
        self.informativeness        = float(np.clip(self.informativeness,        0.0, 1.0))
        self.regional_left          = float(np.clip(self.regional_left,          0.0, 1.0))
        self.regional_center        = float(np.clip(self.regional_center,        0.0, 1.0))
        self.regional_right         = float(np.clip(self.regional_right,         0.0, 1.0))
        self.danger_pixel_fraction  = float(np.clip(self.danger_pixel_fraction,  0.0, 1.0))

    def to_visual_reliability(self) -> VisualReliabilityInput:
        return VisualReliabilityInput(
            global_confidence      = self.global_confidence,
            informativeness        = self.informativeness,
            tier                   = self.tier,
            scale_coherence_small  = self.scale_coherence_small,
            scale_coherence_normal = self.scale_coherence_normal,
            scale_coherence_large  = self.scale_coherence_large,
        )


@dataclass
class LidarPipelineOutput:
    """
    Everything the LiDAR multi-echo pipeline produces per scan.
    """
    p_surface            : float
    p_smoke              : float
    p_unknown            : float
    spatial_consistency  : float
    valid_echo_fraction  : float
    beta_estimate        : float = 0.0
    obstacle_proximity   : float = 0.0   # 0=far/open, 1=very close/blocking

    def __post_init__(self):
        self.p_surface           = float(np.clip(self.p_surface,           0.0, 1.0))
        self.p_smoke             = float(np.clip(self.p_smoke,             0.0, 1.0))
        self.p_unknown           = float(np.clip(self.p_unknown,           0.0, 1.0))
        self.spatial_consistency = float(np.clip(self.spatial_consistency, 0.0, 1.0))
        self.valid_echo_fraction = float(np.clip(self.valid_echo_fraction, 0.0, 1.0))
        self.beta_estimate       = float(max(0.0, self.beta_estimate))
        self.obstacle_proximity  = float(np.clip(self.obstacle_proximity,  0.0, 1.0))

    def to_lidar_reliability(self) -> LidarReliabilityInput:
        return LidarReliabilityInput(
            p_surface            = self.p_surface,
            p_smoke              = self.p_smoke,
            p_unknown            = self.p_unknown,
            spatial_consistency  = self.spatial_consistency,
            valid_echo_fraction  = self.valid_echo_fraction,
            beta_estimate        = self.beta_estimate,
        )


@dataclass
class OdometryInput:
    """
    Body-frame velocity, yaw rate, AND absolute position from onboard
    odometry (PX4/ArduPilot VIO/SLAM estimate, GPS-denied).

    Position fields (x, y, z) were added to support distance_to_last_signal
    tracking in FusionEngine — real flight stacks already expose position
    alongside velocity, so this is received, not derived/integrated here.
    """
    vx       : float = 0.0
    vy       : float = 0.0
    vz       : float = 0.0
    yaw_rate : float = 0.0
    x        : float = 0.0
    y        : float = 0.0
    z        : float = 0.0

    def position(self) -> Tuple[float, float, float]:
        """Convenience accessor for the (x, y, z) tuple."""
        return (self.x, self.y, self.z)


class FusionEngine:

    def __init__(
        self,
        weight_computer          : Optional[SensorWeightComputer] = None,
        thresh                   : Optional[FusionThresholds]     = None,

        # scene_confidence weights
        scene_w_visual           : float = 0.50,
        scene_w_lidar            : float = 0.50,

        # obstacle_confidence weights
        obstacle_w_lidar_surface : float = 0.60,
        obstacle_w_lidar_prox    : float = 0.25,
        obstacle_w_visual_center : float = 0.15,

        # smoke_density weights (fixed — not sensor-weight adjusted)
        smoke_w_beta             : float = 0.50,
        smoke_w_visual_inv       : float = 0.30,
        smoke_w_lidar_smoke      : float = 0.20,

        beta_max                  : float = 1.5,

        # distance_to_last_signal normalization — beyond this distance,
        # the normalized value saturates at 1.0 ("far")
        max_signal_distance       : float = 50.0,
    ):
        self.weight_computer          = weight_computer or SensorWeightComputer()
        self.thresh                   = thresh          or FusionThresholds()

        self.scene_w_visual           = scene_w_visual
        self.scene_w_lidar            = scene_w_lidar

        self.obstacle_w_lidar_surface = obstacle_w_lidar_surface
        self.obstacle_w_lidar_prox    = obstacle_w_lidar_prox
        self.obstacle_w_visual_center = obstacle_w_visual_center

        self.smoke_w_beta             = smoke_w_beta
        self.smoke_w_visual_inv       = smoke_w_visual_inv
        self.smoke_w_lidar_smoke      = smoke_w_lidar_smoke

        self.beta_max                 = beta_max
        self.max_signal_distance      = max_signal_distance

        # stateful fields
        self._danger_timer        : float                           = 0.0
        self._last_nav_state      : NavigationState                 = NavigationState.SAFE
        self._last_timestamp      : float                           = time.time()
        self._last_signal_position: Optional[Tuple[float, float, float]] = None
        self._last_distance_to_signal: float                        = 1.0

    def fuse(
        self,
        gabor      : GaborPipelineOutput,
        lidar      : LidarPipelineOutput,
        odometry   : OdometryInput,
        radar_flag : int            = 0,
        timestamp  : Optional[float] = None,
    ) -> UnifiedPerceptualState:

        ts = timestamp or time.time()
        dt = max(0.0, ts - self._last_timestamp)
        self._last_timestamp = ts

        # 1. sensor weights
        weight_result = self.weight_computer.compute(
            visual = gabor.to_visual_reliability(),
            lidar  = lidar.to_lidar_reliability(),
        )
        w_v = weight_result.w_visual
        w_l = weight_result.w_lidar

        # 2. fuse scene_confidence
        scene_conf = self._fuse_scene_confidence(gabor, lidar, w_v, w_l)

        # 3. fuse obstacle_confidence
        obs_conf = self._fuse_obstacle_confidence(gabor, lidar, w_v, w_l)

        # 4. fuse smoke_density
        smoke = self._fuse_smoke_density(gabor, lidar)

        # 5. regional scores — depth-based, already computed by Gabor pipeline
        reg_l = gabor.regional_left
        reg_c = gabor.regional_center
        reg_r = gabor.regional_right

        # 6. derive dominant sensor
        dominant = UnifiedPerceptualState.derive_dominant_sensor(w_v, w_l, self.thresh)

        # 7. derive nav state (pre-timer update)
        nav_state = UnifiedPerceptualState.derive_navigation_state(
            scene_conf, obs_conf, w_v, w_l, self._danger_timer, self.thresh
        )

        # 8. update danger timer
        self._danger_timer = self._update_danger_timer(nav_state, dt)

        # 9. re-derive nav state with updated timer
        #    timer may have crossed BLIND threshold in this cycle
        nav_state = UnifiedPerceptualState.derive_navigation_state(
            scene_conf, obs_conf, w_v, w_l, self._danger_timer, self.thresh
        )

        # 10. update distance_to_last_signal (tracks UAV position vs. most
        #     recent radar_flag=1 trigger location)
        self._last_distance_to_signal = self._update_signal_distance(
            radar_flag, odometry.position()
        )

        # 11. construct and return UPS
        ups = UnifiedPerceptualState(
            scene_confidence    = scene_conf,
            obstacle_confidence = obs_conf,
            smoke_density       = smoke,
            w_visual            = w_v,
            w_lidar             = w_l,
            regional_left       = reg_l,
            regional_center     = reg_c,
            regional_right      = reg_r,
            radar_flag          = radar_flag,
            danger_timer        = self._danger_timer,
            vx                  = odometry.vx,
            vy                  = odometry.vy,
            vz                  = odometry.vz,
            yaw_rate            = odometry.yaw_rate,
            dominant_sensor     = dominant,
            navigation_state    = nav_state,
            timestamp           = ts,
        )

        self._last_nav_state = nav_state
        return ups

    def get_distance_to_last_signal(self) -> float:
        """
        Most recently computed distance_to_last_signal, normalized [0, 1].
        1.0 = no signal ever detected this mission, or signal is farther
        than max_signal_distance away. Call after fuse().
        """
        return self._last_distance_to_signal

    def _fuse_scene_confidence(
        self,
        gabor : GaborPipelineOutput,
        lidar : LidarPipelineOutput,
        w_v   : float,
        w_l   : float,
    ) -> float:

        visual_score      = gabor.global_confidence
        lidar_scene_score = lidar.p_surface * (0.7 + 0.3 * lidar.spatial_consistency)

        fused = (w_v * visual_score + w_l * lidar_scene_score) / (w_v + w_l + 1e-9)

        # penalty if most pixels are in DANGER tier
        if gabor.danger_pixel_fraction > 0.7:
            fused *= 0.80

        return float(np.clip(fused, 0.0, 1.0))

    def _fuse_obstacle_confidence(
        self,
        gabor : GaborPipelineOutput,
        lidar : LidarPipelineOutput,
        w_v   : float,
        w_l   : float,
    ) -> float:

        lidar_obs  = (self.obstacle_w_lidar_surface * lidar.p_surface +
                      self.obstacle_w_lidar_prox    * lidar.obstacle_proximity)

        visual_obs = 1.0 - gabor.regional_center   # blocked center = obstacle

        lidar_frac  = w_l * (self.obstacle_w_lidar_surface + self.obstacle_w_lidar_prox)
        visual_frac = w_v * self.obstacle_w_visual_center
        total       = lidar_frac + visual_frac + 1e-9

        fused = (lidar_frac * lidar_obs + visual_frac * visual_obs) / total

        return float(np.clip(fused, 0.0, 1.0))

    def _fuse_smoke_density(
        self,
        gabor : GaborPipelineOutput,
        lidar : LidarPipelineOutput,
    ) -> float:

        beta_norm    = float(np.clip(lidar.beta_estimate / self.beta_max, 0.0, 1.0))
        visual_smoke = 1.0 - gabor.informativeness
        lidar_smoke  = lidar.p_smoke

        fused = (self.smoke_w_beta        * beta_norm    +
                 self.smoke_w_visual_inv  * visual_smoke +
                 self.smoke_w_lidar_smoke * lidar_smoke)

        return float(np.clip(fused, 0.0, 1.0))

    def _update_danger_timer(
        self,
        nav_state : NavigationState,
        dt        : float,
    ) -> float:
        """
        Increments during DANGER or BLIND.
        Resets to 0.0 on SAFE or CAUTION.
        """
        if nav_state in (NavigationState.DANGER, NavigationState.BLIND):
            return self._danger_timer + dt
        return 0.0

    def _update_signal_distance(
        self,
        radar_flag       : int,
        current_position : Tuple[float, float, float],
    ) -> float:
        """
        Update last_signal_position if radar just triggered, then compute
        the normalized distance from current position to the most recent
        signal location.

        Returns
        -------
        float — distance_to_last_signal, normalized [0, 1].
                1.0 = no signal ever detected this mission, OR signal is
                farther than max_signal_distance away.
        """
        if radar_flag == 1:
            self._last_signal_position = current_position

        if self._last_signal_position is None:
            return 1.0

        dx = current_position[0] - self._last_signal_position[0]
        dy = current_position[1] - self._last_signal_position[1]
        dz = current_position[2] - self._last_signal_position[2]
        raw_distance = (dx**2 + dy**2 + dz**2) ** 0.5

        return float(min(raw_distance / self.max_signal_distance, 1.0))

    def reset(self):
        """Call between missions or test runs."""
        self._danger_timer            = 0.0
        self._last_nav_state          = NavigationState.SAFE
        self._last_timestamp          = time.time()
        self._last_signal_position    = None
        self._last_distance_to_signal = 1.0 