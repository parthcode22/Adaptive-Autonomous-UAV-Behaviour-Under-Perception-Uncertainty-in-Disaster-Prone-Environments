"""
confidence_map.py — Final Confidence Aggregation Layer
=========================================================
SURE-UAV LiDAR Smoke Filtering Module

Combines outputs from pulse_classifier.py, spatial_filter.py, and
temporal_filter.py into scan-level aggregate scores matching the exact
interface the fusion layer's LidarPipelineOutput expects.

This is the bridge between the LiDAR module and fusion_engine.py.

Aggregation principle: per-point scores are compounded (e.g. p_surface
is boosted by temporal persistence, not just classifier confidence
alone) before being averaged into scan-level scores. A point that looks
clean in one frame but flickers across scans should NOT score as highly
as one that is both clean AND stable.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from .echo_physics import LidarPulse, PulsePhysicsFeatures
from .pulse_classifier import PulseClassification
from .spatial_filter import SpatialPoint, SpatialFilterResult
from .temporal_filter import TemporalPointResult


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class ConfidenceMapConstants:
    """Calibration constants for final confidence aggregation."""

    # Minimum persistence_score multiplier — even a brand-new point
    # (neutral 0.5 persistence) shouldn't be zeroed out entirely
    PERSISTENCE_FLOOR_MULTIPLIER : float = 0.3

    # When spatial_consistency is below this, boost p_smoke for all
    # points in the scan (low consistency = likely scattering medium)
    LOW_SPATIAL_CONSISTENCY_THRESHOLD : float = 0.4
    SPATIAL_SMOKE_BOOST                : float = 0.2

    # obstacle_proximity: distance (meters) considered "very close" (1.0)
    # vs "far" (0.0), for normalizing the closest confident surface point
    PROXIMITY_NEAR_METERS  : float = 0.5
    PROXIMITY_FAR_METERS   : float = 10.0

    # Minimum p_surface (after persistence weighting) to count a point
    # as a valid obstacle candidate for proximity calculation
    MIN_P_SURFACE_FOR_OBSTACLE : float = 0.5


# ---------------------------------------------------------------------------
# Result dataclass — matches LidarPipelineOutput's exact fields
# ---------------------------------------------------------------------------

@dataclass
class ScanConfidenceResult:
    """
    Scan-level aggregate confidence, ready to construct a
    fusion_engine.LidarPipelineOutput directly from these fields.

    Attributes
    ----------
    p_surface            : mean persistence-weighted surface confidence
    p_smoke               : mean smoke confidence (boosted if spatially inconsistent)
    p_unknown             : mean unknown/ambiguous confidence
    spatial_consistency   : passed through from SpatialFilterResult
    valid_echo_fraction   : fraction of pulses with at least one echo
    beta_estimate          : mean extinction coefficient across points
    obstacle_proximity     : normalized proximity of nearest confident obstacle
                            (0=far/none found, 1=very close)
    num_points             : number of points aggregated into this result
    """
    p_surface            : float
    p_smoke               : float
    p_unknown             : float
    spatial_consistency   : float
    valid_echo_fraction   : float
    beta_estimate          : float
    obstacle_proximity     : float
    num_points             : int

    def __post_init__(self):
        self.p_surface           = float(np.clip(self.p_surface,           0.0, 1.0))
        self.p_smoke             = float(np.clip(self.p_smoke,             0.0, 1.0))
        self.p_unknown           = float(np.clip(self.p_unknown,           0.0, 1.0))
        self.spatial_consistency = float(np.clip(self.spatial_consistency, 0.0, 1.0))
        self.valid_echo_fraction = float(np.clip(self.valid_echo_fraction, 0.0, 1.0))
        self.beta_estimate       = float(max(0.0, self.beta_estimate))
        self.obstacle_proximity  = float(np.clip(self.obstacle_proximity,  0.0, 1.0))

    def summary(self) -> str:
        return (
            f"[ScanConfidence] p_surface={self.p_surface:.3f} "
            f"p_smoke={self.p_smoke:.3f} p_unknown={self.p_unknown:.3f} "
            f"spatial={self.spatial_consistency:.3f} "
            f"valid_echo={self.valid_echo_fraction:.3f} "
            f"beta={self.beta_estimate:.3f} "
            f"obstacle_prox={self.obstacle_proximity:.3f} "
            f"(n={self.num_points})"
        )


# ---------------------------------------------------------------------------
# Per-point compounding
# ---------------------------------------------------------------------------

def _compound_point_scores(
    classification     : PulseClassification,
    temporal_result     : TemporalPointResult,
    consts               : ConfidenceMapConstants,
) -> tuple[float, float, float]:
    """
    Compound a single point's classifier + temporal scores into final
    per-point (p_surface, p_smoke, p_unknown).

    p_surface is weighted down if the point doesn't persist over time.
    p_smoke and p_unknown pass through from the classifier unchanged
    at this stage (spatial boost is applied at the scan level instead,
    since spatial_consistency is a scan-wide property, not per-point).
    """
    persistence = temporal_result.persistence_score

    # Floor ensures a single new/unscored point isn't zeroed entirely
    persistence_weight = max(persistence, consts.PERSISTENCE_FLOOR_MULTIPLIER)

    final_p_surface = classification.p_surface * persistence_weight
    final_p_smoke   = classification.p_smoke
    final_p_unknown = classification.p_unknown

    return final_p_surface, final_p_smoke, final_p_unknown


# ---------------------------------------------------------------------------
# Main aggregation entry point
# ---------------------------------------------------------------------------

def compute_scan_confidence(
    pulses              : List[LidarPulse],
    features_list        : List[PulsePhysicsFeatures],
    classifications       : List[PulseClassification],
    spatial_points         : List[SpatialPoint],
    spatial_result         : SpatialFilterResult,
    temporal_results        : List[TemporalPointResult],
    consts                   : ConfidenceMapConstants = ConfidenceMapConstants(),
) -> ScanConfidenceResult:
    """
    Aggregate all per-pulse and per-point layer outputs into a single
    scan-level ScanConfidenceResult.

    Parameters
    ----------
    pulses             : all LidarPulse objects in this scan
    features_list      : PulsePhysicsFeatures, one per pulse, same order as pulses
    classifications    : PulseClassification, one per pulse, same order as pulses
    spatial_points      : SpatialPoint list (only pulses with valid surface echoes)
    spatial_result      : SpatialFilterResult from filter_spatial_consistency()
    temporal_results     : TemporalPointResult list, same order as spatial_points
    consts                : ConfidenceMapConstants (override for tuning)

    Returns
    -------
    ScanConfidenceResult
    """
    # --- valid_echo_fraction: fraction of PULSES with any return at all ---
    if pulses:
        valid_echo_fraction = sum(1 for p in pulses if p.has_return) / len(pulses)
    else:
        valid_echo_fraction = 0.0

    # --- beta_estimate: mean across all pulse features ---
    if features_list:
        beta_estimate = float(np.mean([f.beta_estimate for f in features_list]))
    else:
        beta_estimate = 0.0

    # --- Compound per-point scores (only for points that made it through
    #     spatial_filter, i.e. had a valid best-surface-echo) ---
    compounded_surface = []
    compounded_smoke    = []
    compounded_unknown   = []

    # Build a lookup from spatial_points back to their originating
    # classification, by matching list position (spatial_points is built
    # from the same ordered subset of pulses/classifications upstream)
    for sp, temp_result in zip(spatial_points, temporal_results):
        # Find the classification matching this point's p_surface/p_smoke
        # (spatial_points already carries p_surface/p_smoke copied over
        # from classification at construction time in pulse_to_point())
        dummy_classification = PulseClassification(
            p_surface=sp.p_surface,
            p_smoke=sp.p_smoke,
            p_unknown=0.0,  # not carried on SpatialPoint; see note below
            best_surface_echo_index=0,
            evidence_notes="",
        )
        fp_surface, fp_smoke, fp_unknown = _compound_point_scores(
            dummy_classification, temp_result, consts
        )
        compounded_surface.append(fp_surface)
        compounded_smoke.append(fp_smoke)
        compounded_unknown.append(fp_unknown)

    # Include p_unknown from ALL classifications (not just spatial-filtered
    # subset), since p_unknown often corresponds to points with no valid
    # surface echo at all (e.g. total absorption), which never make it
    # into spatial_points in the first place.
    all_p_unknown = [c.p_unknown for c in classifications]

    p_surface = float(np.mean(compounded_surface)) if compounded_surface else 0.0
    p_smoke   = float(np.mean(compounded_smoke))   if compounded_smoke   else (
        float(np.mean([c.p_smoke for c in classifications])) if classifications else 0.0
    )
    p_unknown = float(np.mean(all_p_unknown)) if all_p_unknown else 0.0

    # --- Spatial consistency boost to p_smoke ---
    spatial_consistency = spatial_result.spatial_consistency
    if spatial_consistency < consts.LOW_SPATIAL_CONSISTENCY_THRESHOLD:
        p_smoke = float(np.clip(p_smoke + consts.SPATIAL_SMOKE_BOOST, 0.0, 1.0))

    # --- obstacle_proximity: closest point with high compounded p_surface ---
    obstacle_proximity = _compute_obstacle_proximity(
        spatial_points, compounded_surface, consts
    )

    return ScanConfidenceResult(
        p_surface            = p_surface,
        p_smoke               = p_smoke,
        p_unknown             = p_unknown,
        spatial_consistency   = spatial_consistency,
        valid_echo_fraction   = valid_echo_fraction,
        beta_estimate          = beta_estimate,
        obstacle_proximity     = obstacle_proximity,
        num_points             = len(spatial_points),
    )


def _compute_obstacle_proximity(
    spatial_points       : List[SpatialPoint],
    compounded_surface     : List[float],
    consts                  : ConfidenceMapConstants,
) -> float:
    """
    Find the closest point with high compounded p_surface and normalize
    its distance into a 0-1 proximity score (1=very close, 0=far/none).
    """
    if not spatial_points or not compounded_surface:
        return 0.0

    candidate_distances = [
        float(np.linalg.norm(sp.as_array()))
        for sp, score in zip(spatial_points, compounded_surface)
        if score >= consts.MIN_P_SURFACE_FOR_OBSTACLE
    ]

    if not candidate_distances:
        return 0.0

    closest = min(candidate_distances)
    near, far = consts.PROXIMITY_NEAR_METERS, consts.PROXIMITY_FAR_METERS

    if closest <= near:
        return 1.0
    if closest >= far:
        return 0.0

    # Linear interpolation between near (1.0) and far (0.0)
    return float(1.0 - (closest - near) / (far - near))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from echo_physics import EchoReturn, extract_pulse_physics
    from pulse_classifier import classify_pulse
    from spatial_filter import pulse_to_point, filter_spatial_consistency
    from temporal_filter import TemporalTracker

    print("=" * 78)
    print("confidence_map.py — Self-Test")
    print("=" * 78)

    rng = np.random.default_rng(3)

    def build_scan(pulse_specs):
        """pulse_specs: list of (azimuth, elevation, [(distance, intensity), ...])"""
        pulses, features_list, classifications, spatial_points = [], [], [], []
        for az, el, echo_specs in pulse_specs:
            echoes = [
                EchoReturn(distance=d, intensity=i, echo_index=idx, total_echoes=len(echo_specs))
                for idx, (d, i) in enumerate(echo_specs)
            ]
            pulse = LidarPulse(echoes=echoes, beam_azimuth=az, beam_elevation=el, timestamp=0.0)
            feats = extract_pulse_physics(pulse)
            cls   = classify_pulse(pulse, feats)
            pt    = pulse_to_point(pulse, cls)

            pulses.append(pulse)
            features_list.append(feats)
            classifications.append(cls)
            if pt:
                spatial_points.append(pt)
        return pulses, features_list, classifications, spatial_points

    # --- Scenario A: Clean wall scan, tracked stable over 3 scans ---
    print("\n[Scenario A: Clean wall, 3 consecutive scans]")
    tracker_a = TemporalTracker()
    for scan_idx in range(3):
        pulse_specs = [
            (az, el, [(6.0 + rng.normal(0, 0.01), 0.4)])
            for az in np.linspace(-0.04, 0.04, 10)
            for el in np.linspace(-0.02, 0.02, 4)
        ]
        pulses, feats, cls, sp_points = build_scan(pulse_specs)
        spatial_result = filter_spatial_consistency(sp_points)
        temporal_results = tracker_a.update(sp_points)

        result = compute_scan_confidence(
            pulses, feats, cls, sp_points, spatial_result, temporal_results
        )
        print(f"  scan {scan_idx}: {result.summary()}")

    # --- Scenario B: Dense smoke scan, total absorption on most pulses ---
    print("\n[Scenario B: Dense smoke, mostly total absorption, 3 scans]")
    tracker_b = TemporalTracker()
    for scan_idx in range(3):
        pulse_specs = []
        for az in np.linspace(-0.04, 0.04, 10):
            for el in np.linspace(-0.02, 0.02, 4):
                if rng.random() < 0.7:
                    pulse_specs.append((az, el, []))  # total absorption
                else:
                    pulse_specs.append((az, el, [(rng.uniform(1.0, 3.0), rng.uniform(0.05, 0.15))]))
        pulses, feats, cls, sp_points = build_scan(pulse_specs)
        spatial_result = filter_spatial_consistency(sp_points)
        temporal_results = tracker_b.update(sp_points)

        result = compute_scan_confidence(
            pulses, feats, cls, sp_points, spatial_result, temporal_results
        )
        print(f"  scan {scan_idx}: {result.summary()}")

    # --- Scenario C: Smoke-then-wall mixed pulses, stable wall behind smoke ---
    print("\n[Scenario C: Smoke-then-wall on every pulse, 3 scans]")
    tracker_c = TemporalTracker()
    for scan_idx in range(3):
        pulse_specs = [
            (az, el, [(2.0 + rng.normal(0, 0.1), 0.05), (9.0 + rng.normal(0, 0.01), 0.50)])
            for az in np.linspace(-0.04, 0.04, 10)
            for el in np.linspace(-0.02, 0.02, 4)
        ]
        pulses, feats, cls, sp_points = build_scan(pulse_specs)
        spatial_result = filter_spatial_consistency(sp_points)
        temporal_results = tracker_c.update(sp_points)

        result = compute_scan_confidence(
            pulses, feats, cls, sp_points, spatial_result, temporal_results
        )
        print(f"  scan {scan_idx}: {result.summary()}")

    print("\n" + "=" * 78)
    print("Expected: A -> high p_surface, low p_smoke, high spatial, close obstacle.")
    print("          B -> low p_surface, high p_smoke, low valid_echo_fraction.")
    print("          C -> high p_surface (wall persists), notable p_smoke (smoke detected).")
    print("=" * 78)