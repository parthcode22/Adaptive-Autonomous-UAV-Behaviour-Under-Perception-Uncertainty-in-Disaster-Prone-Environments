"""
lidar_pipeline.py — LiDAR Module Orchestrator
================================================
SURE-UAV LiDAR Smoke Filtering Module

The single entry point for the entire LiDAR multi-echo smoke filtering
module. Mirrors fusion_pipeline.py's role — ties together all 5 layers
into one clean per-scan call, and exposes a direct conversion to the
exact LidarPipelineOutput interface fusion_engine.py expects.

Pipeline stages, per scan:
    1. echo_physics.extract_pulse_physics()      — per pulse
    2. pulse_classifier.classify_pulse()          — per pulse
    3. spatial_filter.pulse_to_point()             — per pulse -> point
    4. spatial_filter.filter_spatial_consistency() — across the scan
    5. temporal_filter.TemporalTracker.update()    — stateful across scans
    6. confidence_map.compute_scan_confidence()    — final aggregation

This is the only file run_pipeline1.py (or its future LiDAR equivalent)
needs to import once real LiDAR data is available.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import time

from .echo_physics import LidarPulse, PulsePhysicsFeatures, extract_pulse_physics
from .pulse_classifier import PulseClassification, classify_pulse
from .spatial_filter import (
    SpatialPoint,
    SpatialFilterResult,
    SpatialConstants,
    pulse_to_point,
    filter_spatial_consistency,
)
from .temporal_filter import (
    TemporalTracker,
    TemporalConstants,
    TemporalPointResult,
)
from .confidence_map import (
    ScanConfidenceResult,
    ConfidenceMapConstants,
    compute_scan_confidence,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LidarPipelineConfig:
    """
    Runtime configuration for LidarPipeline.

    Attributes
    ----------
    spatial_consts    : SpatialConstants override
    temporal_consts   : TemporalConstants override
    confidence_consts : ConfidenceMapConstants override
    log_to_console    : print scan summary each cycle (dev mode)
    """
    spatial_consts     : SpatialConstants        = None
    temporal_consts     : TemporalConstants        = None
    confidence_consts    : ConfidenceMapConstants    = None
    log_to_console        : bool                       = False

    def __post_init__(self):
        if self.spatial_consts is None:
            self.spatial_consts = SpatialConstants()
        if self.temporal_consts is None:
            self.temporal_consts = TemporalConstants()
        if self.confidence_consts is None:
            self.confidence_consts = ConfidenceMapConstants()


# ---------------------------------------------------------------------------
# LiDAR -> Fusion bridge dataclass
# ---------------------------------------------------------------------------

@dataclass
class LidarToFusionOutput:
    """
    Direct match to fusion_engine.LidarPipelineOutput's constructor
    signature. Build this, then pass its fields straight into
    LidarPipelineOutput(**output.__dict__) in run_pipeline1.py.

    Attributes mirror fusion_engine.LidarPipelineOutput exactly:
    p_surface, p_smoke, p_unknown, spatial_consistency,
    valid_echo_fraction, beta_estimate, obstacle_proximity
    """
    p_surface            : float
    p_smoke               : float
    p_unknown             : float
    spatial_consistency   : float
    valid_echo_fraction   : float
    beta_estimate          : float
    obstacle_proximity     : float


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class LidarPipeline:
    """
    Main orchestrator for the SURE-UAV multi-echo LiDAR smoke filtering
    module. Stateful — holds a TemporalTracker across scan calls.

    Usage
    -----
        pipeline = LidarPipeline()

        # pulses: List[LidarPulse] for one scan (from real/simulated sensor)
        result = pipeline.process_scan(pulses)

        lidar_output = result.to_fusion_output()
        # then: LidarPipelineOutput(**lidar_output.__dict__)
    """

    def __init__(self, config: Optional[LidarPipelineConfig] = None):
        self.config = config or LidarPipelineConfig()

        self._tracker = TemporalTracker(consts=self.config.temporal_consts)

        self._scan_count   : int   = 0
        self._last_result   : Optional[ScanConfidenceResult] = None

    # ------------------------------------------------------------------
    # Public: main per-scan call
    # ------------------------------------------------------------------

    def process_scan(self, pulses: List[LidarPulse]) -> ScanConfidenceResult:
        """
        Run the full 5-layer pipeline on one scan's worth of pulses.

        Parameters
        ----------
        pulses : list of LidarPulse for this scan

        Returns
        -------
        ScanConfidenceResult — also stored internally as get_latest_result()
        """
        # Stage 1 + 2: physics + classification, per pulse
        features_list     : List[PulsePhysicsFeatures] = []
        classifications     : List[PulseClassification]   = []

        for pulse in pulses:
            feats = extract_pulse_physics(pulse)
            cls   = classify_pulse(pulse, feats)
            features_list.append(feats)
            classifications.append(cls)

        # Stage 3: convert to spatial points (only pulses with a valid
        # best-surface-echo produce a point)
        spatial_points: List[SpatialPoint] = []
        for pulse, cls in zip(pulses, classifications):
            pt = pulse_to_point(pulse, cls)
            if pt is not None:
                spatial_points.append(pt)

        # Stage 4: spatial consistency across this scan
        spatial_result = filter_spatial_consistency(
            spatial_points, consts=self.config.spatial_consts
        )

        # Stage 5: temporal persistence (stateful tracker)
        temporal_results: List[TemporalPointResult] = self._tracker.update(spatial_points)

        # Stage 6: final aggregation
        result = compute_scan_confidence(
            pulses             = pulses,
            features_list       = features_list,
            classifications      = classifications,
            spatial_points        = spatial_points,
            spatial_result         = spatial_result,
            temporal_results        = temporal_results,
            consts                   = self.config.confidence_consts,
        )

        self._last_result = result
        self._scan_count  += 1

        if self.config.log_to_console:
            print(f"[LidarPipeline #{self._scan_count:05d}] {result.summary()}")

        return result

    # ------------------------------------------------------------------
    # Public: accessors
    # ------------------------------------------------------------------

    def get_latest_result(self) -> Optional[ScanConfidenceResult]:
        """Most recent ScanConfidenceResult, or None if no scan processed yet."""
        return self._last_result

    def get_scan_count(self) -> int:
        """Number of scans processed so far."""
        return self._scan_count

    # ------------------------------------------------------------------
    # Public: reset
    # ------------------------------------------------------------------

    def reset(self):
        """Full reset — call between missions or test runs."""
        self._tracker.reset()
        self._scan_count  = 0
        self._last_result = None


# ---------------------------------------------------------------------------
# Conversion helper — attach to ScanConfidenceResult via monkeypatch-free
# free function (keeps confidence_map.py untouched)
# ---------------------------------------------------------------------------

def to_fusion_output(result: ScanConfidenceResult) -> LidarToFusionOutput:
    """
    Convert a ScanConfidenceResult into the exact field set
    fusion_engine.LidarPipelineOutput expects.

    Usage in run_pipeline1.py:
        from lidar.lidar_pipeline import LidarPipeline, to_fusion_output
        from fusion_1.fusion_engine import LidarPipelineOutput

        result = lidar_pipeline.process_scan(pulses)
        bridge = to_fusion_output(result)
        lidar_out = LidarPipelineOutput(
            p_surface=bridge.p_surface,
            p_smoke=bridge.p_smoke,
            p_unknown=bridge.p_unknown,
            spatial_consistency=bridge.spatial_consistency,
            valid_echo_fraction=bridge.valid_echo_fraction,
            beta_estimate=bridge.beta_estimate,
            obstacle_proximity=bridge.obstacle_proximity,
        )
    """
    return LidarToFusionOutput(
        p_surface            = result.p_surface,
        p_smoke               = result.p_smoke,
        p_unknown             = result.p_unknown,
        spatial_consistency   = result.spatial_consistency,
        valid_echo_fraction   = result.valid_echo_fraction,
        beta_estimate          = result.beta_estimate,
        obstacle_proximity     = result.obstacle_proximity,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from echo_physics import EchoReturn
    import numpy as np

    print("=" * 78)
    print("lidar_pipeline.py — Self-Test")
    print("=" * 78)

    rng = np.random.default_rng(11)

    def build_wall_scan(distance=6.0, noise=0.01):
        pulses = []
        for az in np.linspace(-0.04, 0.04, 10):
            for el in np.linspace(-0.02, 0.02, 4):
                d = distance + rng.normal(0, noise)
                echoes = [EchoReturn(distance=d, intensity=0.4, echo_index=0, total_echoes=1)]
                pulses.append(LidarPulse(echoes=echoes, beam_azimuth=az, beam_elevation=el, timestamp=time.time()))
        return pulses

    def build_smoke_scan():
        pulses = []
        for az in np.linspace(-0.04, 0.04, 10):
            for el in np.linspace(-0.02, 0.02, 4):
                if rng.random() < 0.6:
                    echoes = []  # total absorption
                else:
                    d = rng.uniform(1.0, 3.0)
                    i = rng.uniform(0.05, 0.15)
                    echoes = [EchoReturn(distance=d, intensity=i, echo_index=0, total_echoes=1)]
                pulses.append(LidarPulse(echoes=echoes, beam_azimuth=az, beam_elevation=el, timestamp=time.time()))
        return pulses

    pipeline = LidarPipeline(config=LidarPipelineConfig(log_to_console=False))

    print("\n[Test 1: Clean wall, 4 consecutive scans]")
    for i in range(4):
        result = pipeline.process_scan(build_wall_scan())
        print(f"  scan {i}: {result.summary()}")

    print(f"\n  Fusion bridge output (scan 4): {to_fusion_output(result)}")

    pipeline.reset()
    print("\n[Test 2: Dense smoke, 4 consecutive scans, after reset()]")
    for i in range(4):
        result = pipeline.process_scan(build_smoke_scan())
        print(f"  scan {i}: {result.summary()}")

    print(f"\n  Fusion bridge output (scan 4): {to_fusion_output(result)}")

    print(f"\n  Total scans processed: {pipeline.get_scan_count()}")
    print("\n" + "=" * 78)
    print("Self-test complete.")
    print("=" * 78)