"""
fusion_pipeline.py — Fusion Pipeline Orchestrator
==================================================
SURE-UAV Fusion Layer

Top-level entry point for the entire fusion system.
This is the only file run_pipeline1.py needs to import.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional
import numpy as np
import time

from .ups_vector import (
    UnifiedPerceptualState,
    NavigationState,
    make_blind_ups,
)
from .sensor_weights import SensorWeightComputer
from .fusion_engine import (
    FusionEngine,
    GaborPipelineOutput,
    LidarPipelineOutput,
    OdometryInput,
)
@dataclass
class FusionPipelineConfig:
    """
    Runtime configuration for FusionPipeline.

    Attributes
    ----------
    history_length    : how many past UPS vectors to retain for trend analysis
    fallback_on_error : return BLIND UPS on exception instead of crashing
    log_to_console    : print UPS summary each cycle (dev mode)
    lidar_available   : False = LiDAR not yet integrated, use neutral fallback
    gabor_available   : False = visual pipeline not running, use low-conf fallback
    """
    history_length    : int  = 30
    fallback_on_error : bool = True
    log_to_console    : bool = False
    lidar_available   : bool = True
    gabor_available   : bool = True


class FusionPipeline:

    def __init__(
        self,
        config : Optional[FusionPipelineConfig] = None,
        engine : Optional[FusionEngine]         = None,
    ):
        self.config = config or FusionPipelineConfig()
        self.engine = engine or FusionEngine()

        # rolling UPS history
        self._history : Deque[UnifiedPerceptualState] = deque(
            maxlen=self.config.history_length
        )

        # cycle tracking
        self._cycle_count   : int   = 0
        self._last_cycle_ts : float = time.time()
        self._fps_estimate  : float = 0.0

        # always holds the most recent UPS — safe to read between update() calls
        self._latest_ups : UnifiedPerceptualState = make_blind_ups(
            timestamp=time.time()
        )
    def update(
        self,
        gabor      : Optional[GaborPipelineOutput] = None,
        lidar      : Optional[LidarPipelineOutput] = None,
        odometry   : Optional[OdometryInput]       = None,
        radar_flag : int                           = 0,
        timestamp  : Optional[float]               = None,
    ) -> UnifiedPerceptualState:
        """
        Run one fusion cycle. Call this once per frame from run_pipeline1.py.

        Missing sensors are handled gracefully via fallbacks — the pipeline
        never crashes due to an unavailable sensor.
        """
        ts = timestamp or time.time()

        try:
            gabor_in = gabor    or self._fallback_gabor()
            lidar_in = lidar    or self._fallback_lidar()
            odom_in  = odometry or OdometryInput()

            ups = self.engine.fuse(
                gabor      = gabor_in,
                lidar      = lidar_in,
                odometry   = odom_in,
                radar_flag = radar_flag,
                timestamp  = ts,
            )

        except Exception as e:
            if self.config.fallback_on_error:
                print(f"[FusionPipeline] ERROR: {e} — returning BLIND UPS")
                ups = make_blind_ups(timestamp=ts, radar_flag=radar_flag)
            else:
                raise

        # store and update
        self._history.append(ups)
        self._latest_ups = ups
        self._update_fps(ts)
        self._cycle_count += 1

        if self.config.log_to_console:
            print(f"[Fusion #{self._cycle_count:05d}] {ups.summary()}")

        return ups
    def get_ppo_state(self) -> np.ndarray:
        """14-dimensional PPO state vector from the latest fusion cycle."""
        return self._latest_ups.to_ppo_state()

    def get_latest_ups(self) -> UnifiedPerceptualState:
        """Latest UPS object — full access to all fields."""
        return self._latest_ups

    def get_display_metrics(self) -> Dict:
        """
        Flat dictionary for the 2x2 pipeline display panel.
        All values are floats or strings — ready for cv2.putText.
        """
        ups = self._latest_ups
        return {
            # navigation
            "nav_state"           : ups.navigation_state.value,
            "dominant_sensor"     : ups.dominant_sensor.value,
            # confidence
            "scene_confidence"    : round(ups.scene_confidence,    3),
            "obstacle_confidence" : round(ups.obstacle_confidence, 3),
            "smoke_density"       : round(ups.smoke_density,       3),
            # weights
            "w_visual"            : round(ups.w_visual,            3),
            "w_lidar"             : round(ups.w_lidar,             3),
            # regional
            "regional_left"       : round(ups.regional_left,       3),
            "regional_center"     : round(ups.regional_center,     3),
            "regional_right"      : round(ups.regional_right,      3),
            # external
            "radar_flag"          : ups.radar_flag,
            "danger_timer"        : round(ups.danger_timer,        1),
            # odometry
            "vx"                  : round(ups.vx,                  2),
            "vy"                  : round(ups.vy,                  2),
            "vz"                  : round(ups.vz,                  2),
            # meta
            "fusion_fps"          : round(self._fps_estimate,      1),
            "fusion_cycle"        : self._cycle_count,
        }
    def get_scene_confidence_trend(self, window: int = 15) -> float:
        """
        Slope of scene_confidence over last `window` cycles.
        Positive = improving. Negative = degrading.
        Action Selector uses this to slow down BEFORE hitting DANGER.
        """
        if len(self._history) < 2:
            return 0.0
        recent = list(self._history)[-window:]
        scores = [u.scene_confidence for u in recent]
        x      = np.arange(len(scores), dtype=float)
        return float(np.polyfit(x, scores, 1)[0])

    def get_smoke_density_trend(self, window: int = 15) -> float:
        """
        Slope of smoke_density over last `window` cycles.
        Positive = smoke getting worse. Negative = clearing.
        """
        if len(self._history) < 2:
            return 0.0
        recent = list(self._history)[-window:]
        scores = [u.smoke_density for u in recent]
        x      = np.arange(len(scores), dtype=float)
        return float(np.polyfit(x, scores, 1)[0])

    def get_sustained_nav_state(self, window: int = 10) -> NavigationState:
        """
        Most common NavigationState over last `window` cycles.
        More stable than reading latest UPS directly — filters single-frame spikes.
        """
        if not self._history:
            return NavigationState.BLIND
        recent = list(self._history)[-window:]
        counts : Dict[NavigationState, int] = {}
        for ups in recent:
            counts[ups.navigation_state] = counts.get(ups.navigation_state, 0) + 1
        return max(counts, key=lambda s: counts[s])

    def is_scene_degrading(self, window: int = 15, threshold: float = -0.01) -> bool:
        """
        True if scene_confidence has been consistently falling.
        -0.01 per frame = scene degrading fast enough to act on.
        """
        return self.get_scene_confidence_trend(window) < threshold

    def history_as_array(self) -> np.ndarray:
        """
        Full UPS history as (N, 14) numpy array of PPO states.
        Used for offline analysis and training data collection.
        """
        if not self._history:
            return np.zeros((0, 14), dtype=np.float32)
        return np.stack([u.to_ppo_state() for u in self._history], axis=0)
    
    def _fallback_gabor(self) -> GaborPipelineOutput:
        """
        Used when visual pipeline is unavailable.
        Low confidence → LiDAR will dominate weights automatically.
        """
        return GaborPipelineOutput(
            global_confidence      = 0.10,
            informativeness        = 0.08,
            tier                   = 'DANGER',
            scale_coherence_small  = 0.20,
            scale_coherence_normal = 0.18,
            scale_coherence_large  = 0.15,
            regional_left          = 0.33,
            regional_center        = 0.33,
            regional_right         = 0.33,
            danger_pixel_fraction  = 0.80,
        )

    def _fallback_lidar(self) -> LidarPipelineOutput:
        """
        Used when LiDAR module is not yet integrated (current dev phase).
        Neutral scores → visual will dominate weights automatically.
        """
        return LidarPipelineOutput(
            p_surface            = 0.30,
            p_smoke              = 0.40,
            p_unknown            = 0.30,
            spatial_consistency  = 0.30,
            valid_echo_fraction  = 0.40,
            beta_estimate        = 0.20,
            obstacle_proximity   = 0.20,
        )

    def _update_fps(self, ts: float):
        dt = ts - self._last_cycle_ts
        self._last_cycle_ts = ts
        if dt > 0:
            instant = 1.0 / dt
            alpha   = 0.1
            self._fps_estimate = (
                alpha * instant + (1.0 - alpha) * self._fps_estimate
            ) if self._fps_estimate > 0 else instant

    def reset(self):
        """Full reset. Call between missions."""
        self._history.clear()
        self._cycle_count   = 0
        self._last_cycle_ts = time.time()
        self._fps_estimate  = 0.0
        self._latest_ups    = make_blind_ups(timestamp=time.time())
        self.engine.reset()
    