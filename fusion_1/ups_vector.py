"""
ups_vector.py — Unified Perceptual State (UPS) Vector
======================================================
SURE-UAV Fusion Layer | Option B: Parallel Weighted Fusion

Defines the UPS dataclass that serves as the single unified output of the
fusion layer. This is what the PPO agent consumes — not raw Gabor scores
or raw LiDAR confidences separately.

Navigation State Logic:
    SAFE    → scene_confidence ≥ 0.7 AND obstacle_confidence < 0.4
    CAUTION → scene_confidence 0.4–0.7 OR obstacle_confidence 0.4–0.7
    DANGER  → scene_confidence < 0.4 AND obstacle_confidence ≥ 0.4
    BLIND   → scene_confidence < 0.2 AND both sensors degraded

PPO State Vector (14-dimensional):
    [scene_confidence, obstacle_confidence, smoke_density,
     w_visual, w_lidar,
     regional_left, regional_center, regional_right,
     radar_flag, danger_timer,
     vx, vy, vz, yaw_rate]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DominantSensor(Enum):
    """Which sensor is currently carrying the most perceptual weight."""
    VISUAL = "VISUAL"       # Gabor/camera dominates (w_visual > 0.6)
    LIDAR  = "LIDAR"        # LiDAR dominates (w_lidar > 0.6)
    FUSED  = "FUSED"        # Both sensors contributing meaningfully
    NONE   = "NONE"         # Both sensors degraded — BLIND state


class NavigationState(Enum):
    """
    Discrete navigation state derived from the UPS vector.
    Maps directly to PPO action space constraints.
    """
    SAFE    = "SAFE"      # Full speed forward permitted
    CAUTION = "CAUTION"   # Reduced speed, heightened sensing
    DANGER  = "DANGER"    # Immediate brake / turn decision required
    BLIND   = "BLIND"     # Both sensors failed → BACKTRACK + UWB only


# ---------------------------------------------------------------------------
# Thresholds (tunable without changing logic)
# ---------------------------------------------------------------------------

class FusionThresholds:
    """
    Centralised threshold constants for navigation state derivation.
    Modify here only — never hardcode elsewhere.
    """
    # scene_confidence thresholds
    SCENE_SAFE_MIN       : float = 0.70   # above this → potentially SAFE
    SCENE_CAUTION_MIN    : float = 0.40   # 0.40–0.70 → CAUTION band
    SCENE_BLIND_MAX      : float = 0.20   # below this → BLIND candidate

    # obstacle_confidence thresholds
    OBS_SAFE_MAX         : float = 0.40   # below this → no blocking obstacle
    OBS_DANGER_MIN       : float = 0.40   # above this → obstacle confirmed
    OBS_CAUTION_MIN      : float = 0.40   # overlaps danger — checked with scene

    # smoke_density thresholds
    SMOKE_HEAVY          : float = 0.70   # heavy smoke flag
    SMOKE_MODERATE       : float = 0.40   # moderate smoke flag

    # sensor weight thresholds for dominant_sensor classification
    SENSOR_DOMINANT_MIN  : float = 0.60   # weight above this = dominant

    # danger timer (seconds of sustained DANGER before BLIND escalation)
    DANGER_TIMER_BLIND   : float = 5.0

    # minimum sensor weight floor (prevents division-by-zero and full dropout)
    WEIGHT_FLOOR         : float = 0.05


# ---------------------------------------------------------------------------
# UPS Dataclass
# ---------------------------------------------------------------------------

@dataclass
class UnifiedPerceptualState:
    """
    The single unified output of the SURE-UAV fusion layer.

    All fields are normalised to [0, 1] unless stated otherwise.
    This object is the input interface to the PPO agent.

    Attributes
    ----------
    scene_confidence : float
        Overall perceptual quality. High = sensors agree the scene is
        readable. Low = visual/LiDAR both reporting degraded perception.

    obstacle_confidence : float
        Probability that a solid obstacle exists in the current flight path.
        Derived from LiDAR p_surface weighted by sensor trust.

    smoke_density : float
        Estimated smoke/dust density in the environment.
        Derived from LiDAR β_estimate and Gabor informativeness inversion.

    w_visual : float
        Dynamic weight assigned to the visual (Gabor) sensor this cycle.

    w_lidar : float
        Dynamic weight assigned to the LiDAR sensor this cycle.
        Note: w_visual + w_lidar = 1.0 always.

    regional_left : float
        Depth-based openness score for left flight sector (0=blocked, 1=open).

    regional_center : float
        Depth-based openness score for center flight sector.

    regional_right : float
        Depth-based openness score for right flight sector.

    radar_flag : int
        UWB radar trigger: 1 = respiration signature detected, 0 = none.

    danger_timer : float
        Seconds elapsed in current sustained DANGER/BLIND state.
        Resets to 0.0 on transition to SAFE or CAUTION.

    vx, vy, vz : float
        UAV body-frame velocity components from odometry (m/s).

    yaw_rate : float
        UAV yaw rate from odometry (rad/s).

    dominant_sensor : DominantSensor
        Which sensor is currently driving perception decisions.

    navigation_state : NavigationState
        Discrete navigation state for PPO action masking / reward shaping.

    timestamp : float
        System time at UPS creation (seconds). Used by temporal filter.
    """

    # --- Core perceptual fields ---
    scene_confidence    : float = 0.0
    obstacle_confidence : float = 0.0
    smoke_density       : float = 0.0

    # --- Sensor weights ---
    w_visual            : float = 0.5
    w_lidar             : float = 0.5

    # --- Regional navigation scores (from depth map) ---
    regional_left       : float = 0.5
    regional_center     : float = 0.5
    regional_right      : float = 0.5

    # --- External sensor flags ---
    radar_flag          : int   = 0
    danger_timer        : float = 0.0

    # --- Odometry ---
    vx                  : float = 0.0
    vy                  : float = 0.0
    vz                  : float = 0.0
    yaw_rate            : float = 0.0

    # --- Derived state (computed post-init) ---
    dominant_sensor     : DominantSensor   = field(default=DominantSensor.NONE)
    navigation_state    : NavigationState  = field(default=NavigationState.BLIND)

    # --- Metadata ---
    timestamp           : float = 0.0

    def __post_init__(self):
        """Clamp all float fields to valid ranges after construction."""
        self.scene_confidence    = float(np.clip(self.scene_confidence,    0.0, 1.0))
        self.obstacle_confidence = float(np.clip(self.obstacle_confidence, 0.0, 1.0))
        self.smoke_density       = float(np.clip(self.smoke_density,       0.0, 1.0))
        self.w_visual            = float(np.clip(self.w_visual,            0.0, 1.0))
        self.w_lidar             = float(np.clip(self.w_lidar,             0.0, 1.0))
        self.regional_left       = float(np.clip(self.regional_left,       0.0, 1.0))
        self.regional_center     = float(np.clip(self.regional_center,     0.0, 1.0))
        self.regional_right      = float(np.clip(self.regional_right,      0.0, 1.0))
        self.danger_timer        = float(max(0.0, self.danger_timer))

    # ------------------------------------------------------------------
    # Navigation State Derivation
    # ------------------------------------------------------------------

    @staticmethod
    def derive_navigation_state(
        scene_conf      : float,
        obstacle_conf   : float,
        w_visual        : float,
        w_lidar         : float,
        danger_timer    : float,
        thresh          : FusionThresholds = FusionThresholds()
    ) -> NavigationState:
        """
        Derive the discrete NavigationState from UPS scalar fields.

        Priority order (evaluated top-down, first match wins):
            1. BLIND  — both sensors critically degraded OR sustained danger
            2. DANGER — obstacle confirmed OR scene unreadable
            3. CAUTION — intermediate confidence band
            4. SAFE   — high scene confidence, no obstacle

        Parameters
        ----------
        scene_conf    : overall scene confidence (0–1)
        obstacle_conf : obstacle presence probability (0–1)
        w_visual      : current visual sensor weight
        w_lidar       : current LiDAR sensor weight
        danger_timer  : seconds in sustained non-SAFE state
        thresh        : FusionThresholds instance

        Returns
        -------
        NavigationState enum value
        """
        both_sensors_degraded = (
            w_visual < thresh.WEIGHT_FLOOR + 0.05 and
            w_lidar  < thresh.WEIGHT_FLOOR + 0.05
        )

        # Rule 1: BLIND
        if (scene_conf < thresh.SCENE_BLIND_MAX and both_sensors_degraded) or \
           (danger_timer >= thresh.DANGER_TIMER_BLIND):
            return NavigationState.BLIND

        # Rule 2: DANGER
        if scene_conf < thresh.SCENE_CAUTION_MIN and \
           obstacle_conf >= thresh.OBS_DANGER_MIN:
            return NavigationState.DANGER

        # Also DANGER if scene is very unreadable even without confirmed obstacle
        if scene_conf < thresh.SCENE_BLIND_MAX + 0.05:
            return NavigationState.DANGER

        # Rule 3: CAUTION
        in_caution_scene = thresh.SCENE_CAUTION_MIN <= scene_conf < thresh.SCENE_SAFE_MIN
        obstacle_moderate = thresh.OBS_CAUTION_MIN <= obstacle_conf < thresh.OBS_DANGER_MIN + 0.2

        if in_caution_scene or obstacle_moderate:
            return NavigationState.CAUTION

        # Rule 4: SAFE
        if scene_conf >= thresh.SCENE_SAFE_MIN and \
           obstacle_conf < thresh.OBS_SAFE_MAX:
            return NavigationState.SAFE

        # Default fallback
        return NavigationState.CAUTION

    @staticmethod
    def derive_dominant_sensor(
        w_visual : float,
        w_lidar  : float,
        thresh   : FusionThresholds = FusionThresholds()
    ) -> DominantSensor:
        """
        Classify which sensor is currently dominant based on weights.

        Parameters
        ----------
        w_visual : visual sensor weight
        w_lidar  : LiDAR sensor weight
        thresh   : FusionThresholds instance

        Returns
        -------
        DominantSensor enum value
        """
        both_low = (
            w_visual < thresh.WEIGHT_FLOOR + 0.05 and
            w_lidar  < thresh.WEIGHT_FLOOR + 0.05
        )
        if both_low:
            return DominantSensor.NONE

        if w_visual >= thresh.SENSOR_DOMINANT_MIN:
            return DominantSensor.VISUAL

        if w_lidar >= thresh.SENSOR_DOMINANT_MIN:
            return DominantSensor.LIDAR

        return DominantSensor.FUSED

    # ------------------------------------------------------------------
    # PPO Interface
    # ------------------------------------------------------------------

    def to_ppo_state(self) -> np.ndarray:
        """
        Serialise UPS to the 14-dimensional PPO state vector.

        Vector layout:
            [0]  scene_confidence
            [1]  obstacle_confidence
            [2]  smoke_density
            [3]  w_visual
            [4]  w_lidar
            [5]  regional_left
            [6]  regional_center
            [7]  regional_right
            [8]  radar_flag          (float cast of int 0/1)
            [9]  danger_timer        (normalised: clipped at 10s → 0–1)
            [10] vx                  (clipped ±5 m/s → normalised 0–1)
            [11] vy
            [12] vz
            [13] yaw_rate            (clipped ±π rad/s → normalised 0–1)

        Returns
        -------
        np.ndarray of shape (14,), dtype float32
        """
        danger_timer_norm = float(np.clip(self.danger_timer / 10.0, 0.0, 1.0))
        vx_norm           = float(np.clip((self.vx + 5.0) / 10.0,  0.0, 1.0))
        vy_norm           = float(np.clip((self.vy + 5.0) / 10.0,  0.0, 1.0))
        vz_norm           = float(np.clip((self.vz + 5.0) / 10.0,  0.0, 1.0))
        yaw_norm          = float(np.clip((self.yaw_rate + np.pi) / (2 * np.pi), 0.0, 1.0))

        return np.array([
            self.scene_confidence,
            self.obstacle_confidence,
            self.smoke_density,
            self.w_visual,
            self.w_lidar,
            self.regional_left,
            self.regional_center,
            self.regional_right,
            float(self.radar_flag),
            danger_timer_norm,
            vx_norm,
            vy_norm,
            vz_norm,
            yaw_norm,
        ], dtype=np.float32)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable one-line summary for logging/display."""
        return (
            f"[UPS] nav={self.navigation_state.value:<8} "
            f"scene={self.scene_confidence:.3f} "
            f"obs={self.obstacle_confidence:.3f} "
            f"smoke={self.smoke_density:.3f} "
            f"w=[{self.w_visual:.2f}v/{self.w_lidar:.2f}l] "
            f"dom={self.dominant_sensor.value:<6} "
            f"reg=[L{self.regional_left:.2f}|C{self.regional_center:.2f}|R{self.regional_right:.2f}] "
            f"radar={self.radar_flag} "
            f"dtimer={self.danger_timer:.1f}s"
        )

    def to_dict(self) -> dict:
        """Serialise to dictionary for logging / JSON export."""
        return {
            "scene_confidence"    : round(self.scene_confidence,    4),
            "obstacle_confidence" : round(self.obstacle_confidence, 4),
            "smoke_density"       : round(self.smoke_density,       4),
            "w_visual"            : round(self.w_visual,            4),
            "w_lidar"             : round(self.w_lidar,             4),
            "regional_left"       : round(self.regional_left,       4),
            "regional_center"     : round(self.regional_center,     4),
            "regional_right"      : round(self.regional_right,      4),
            "radar_flag"          : self.radar_flag,
            "danger_timer"        : round(self.danger_timer,        2),
            "vx"                  : round(self.vx,                  4),
            "vy"                  : round(self.vy,                  4),
            "vz"                  : round(self.vz,                  4),
            "yaw_rate"            : round(self.yaw_rate,            4),
            "dominant_sensor"     : self.dominant_sensor.value,
            "navigation_state"    : self.navigation_state.value,
            "timestamp"           : round(self.timestamp,           4),
        }


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_blind_ups(
    danger_timer : float = 0.0,
    timestamp    : float = 0.0,
    radar_flag   : int   = 0,
) -> UnifiedPerceptualState:
    """
    Construct a UPS representing a fully BLIND/degraded state.
    Used as a safe default when fusion fails or sensors are unavailable.
    """
    thresh = FusionThresholds()
    ups = UnifiedPerceptualState(
        scene_confidence    = 0.0,
        obstacle_confidence = 0.5,   # assume possible obstacle when blind
        smoke_density       = 1.0,
        w_visual            = thresh.WEIGHT_FLOOR,
        w_lidar             = thresh.WEIGHT_FLOOR,
        regional_left       = 0.1,
        regional_center     = 0.1,
        regional_right      = 0.1,
        radar_flag          = radar_flag,
        danger_timer        = danger_timer,
        dominant_sensor     = DominantSensor.NONE,
        navigation_state    = NavigationState.BLIND,
        timestamp           = timestamp,
    )
    return ups


def make_safe_ups(timestamp: float = 0.0) -> UnifiedPerceptualState:
    """
    Construct a UPS representing a fully clear, safe environment.
    Used for unit testing and simulation warm-up.
    """
    ups = UnifiedPerceptualState(
        scene_confidence    = 0.95,
        obstacle_confidence = 0.05,
        smoke_density       = 0.02,
        w_visual            = 0.55,
        w_lidar             = 0.45,
        regional_left       = 0.90,
        regional_center     = 0.92,
        regional_right      = 0.88,
        radar_flag          = 0,
        danger_timer        = 0.0,
        dominant_sensor     = DominantSensor.FUSED,
        navigation_state    = NavigationState.SAFE,
        timestamp           = timestamp,
    )
    return ups


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("UPS Vector Self-Test")
    print("=" * 70)

    thresh = FusionThresholds()

    test_cases = [
        # (label, scene_conf, obs_conf, w_v, w_l, danger_timer)
        ("SAFE scenario",           0.85, 0.10, 0.55, 0.45, 0.0),
        ("CAUTION — mid scene",     0.55, 0.35, 0.50, 0.50, 0.0),
        ("DANGER — smoke+obstacle", 0.30, 0.65, 0.30, 0.30, 0.0),
        ("BLIND — both degraded",   0.10, 0.50, 0.06, 0.06, 0.0),
        ("BLIND — timer expired",   0.45, 0.30, 0.40, 0.40, 6.5),
    ]

    for label, sc, oc, wv, wl, dt in test_cases:
        nav   = UnifiedPerceptualState.derive_navigation_state(sc, oc, wv, wl, dt, thresh)
        dom   = UnifiedPerceptualState.derive_dominant_sensor(wv, wl, thresh)
        smoke = 1.0 - sc   # simple proxy for self-test

        ups = UnifiedPerceptualState(
            scene_confidence    = sc,
            obstacle_confidence = oc,
            smoke_density       = smoke,
            w_visual            = wv,
            w_lidar             = wl,
            regional_left       = 0.7,
            regional_center     = 0.5,
            regional_right      = 0.8,
            radar_flag          = 0,
            danger_timer        = dt,
            dominant_sensor     = dom,
            navigation_state    = nav,
            timestamp           = time.time(),
        )

        print(f"\n  [{label}]")
        print(f"  {ups.summary()}")
        ppo = ups.to_ppo_state()
        print(f"  PPO vector (14d): {np.round(ppo, 3)}")

    print("\n  --- Factory helpers ---")
    print(f"  BLIND UPS: {make_blind_ups(danger_timer=3.0).summary()}")
    print(f"  SAFE  UPS: {make_safe_ups().summary()}")
    print("\n  Self-test complete.")