"""
spatial_filter.py — Spatial Consistency Filter
================================================
SURE-UAV LiDAR Smoke Filtering Module

Takes multiple pulses from a single scan and checks whether their best
surface-echo points form a geometrically consistent plane (RANSAC) or
a spatially chaotic cluster (smoke/dust).

A real wall produces points that fit a plane well.
Smoke produces points that scatter incoherently in 3D space, even if
individual pulses showed decent p_surface from pulse_classifier.py.

This is the layer that catches what pulse_classifier.py CAN'T see —
single-pulse reasoning has no way to know if its neighbours agree.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from echo_physics import LidarPulse, PulsePhysicsFeatures
from pulse_classifier import PulseClassification


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class SpatialConstants:
    """Calibration constants for RANSAC plane fitting and clustering."""

    # Minimum points required to attempt plane fitting.
    # Below ~15 points, RANSAC's 3-point sampling can accidentally fit
    # a plane through pure-chance noise (small-sample coincidence).
    # Real LiDAR scans have hundreds/thousands of points, so this floor
    # only matters for small test scans or sparse regions.
    MIN_POINTS_FOR_PLANE     : int   = 15

    # RANSAC: distance threshold for a point to count as a plane inlier (meters)
    # Real surfaces vary by mm-scale sensor noise; smoke/dust candidate
    # points (even after p_surface filtering) vary by 10s of cm because
    # each pulse is hitting an independent, unrelated scattering event.
    RANSAC_INLIER_THRESHOLD  : float = 0.05

    # RANSAC: number of random sampling iterations
    RANSAC_ITERATIONS        : int   = 200

    # RANSAC: minimum inlier fraction to consider the plane fit meaningful
    MIN_INLIER_FRACTION      : float = 0.6

    # Only consider points with p_surface above this as plane-fit candidates
    MIN_P_SURFACE_FOR_FIT    : float = 0.3


# ---------------------------------------------------------------------------
# Point representation
# ---------------------------------------------------------------------------

@dataclass
class SpatialPoint:
    """
    A single 3D point derived from a pulse's best surface echo.

    Attributes
    ----------
    x, y, z    : Cartesian coordinates (meters), sensor-frame
    p_surface  : carried over from PulseClassification
    p_smoke    : carried over from PulseClassification
    pulse_id   : optional identifier back to the source pulse
    """
    x         : float
    y         : float
    z         : float
    p_surface : float
    p_smoke   : float
    pulse_id  : Optional[int] = None

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SpatialFilterResult:
    """
    Output of the spatial consistency filter for one scan.

    Attributes
    ----------
    spatial_consistency : float (0-1) — fraction of candidate points that
                           fit a single dominant plane (RANSAC inlier ratio)
    plane_found          : whether a meaningful plane was detected at all
    plane_normal         : unit normal vector of the best-fit plane, or None
    inlier_indices       : indices into the input point list that are inliers
    outlier_indices      : indices that are outliers (likely smoke/noise)
    num_candidate_points : how many points had high enough p_surface to be
                           considered for plane fitting
    """
    spatial_consistency  : float
    plane_found          : bool
    plane_normal         : Optional[np.ndarray]
    inlier_indices        : List[int]
    outlier_indices       : List[int]
    num_candidate_points  : int

    def __post_init__(self):
        self.spatial_consistency = float(np.clip(self.spatial_consistency, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def pulse_to_point(
    pulse          : LidarPulse,
    classification : PulseClassification,
) -> Optional[SpatialPoint]:
    """
    Convert a pulse's best surface echo into a 3D SpatialPoint using
    spherical-to-Cartesian conversion from beam geometry + echo distance.

    Returns None if the pulse has no valid surface echo to convert
    (best_surface_echo_index == -1, i.e. total absorption).

    Parameters
    ----------
    pulse          : original LidarPulse (for beam_azimuth/elevation)
    classification : PulseClassification (for best_surface_echo_index, p_surface)

    Returns
    -------
    SpatialPoint or None
    """
    idx = classification.best_surface_echo_index
    if idx < 0 or idx >= len(pulse.echoes):
        return None

    echo = pulse.echoes[idx]
    r    = echo.distance
    az   = pulse.beam_azimuth
    el   = pulse.beam_elevation

    # Standard spherical -> Cartesian (sensor-frame, x=forward, y=right, z=up)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)

    return SpatialPoint(
        x=float(x), y=float(y), z=float(z),
        p_surface=classification.p_surface,
        p_smoke=classification.p_smoke,
        pulse_id=pulse.pulse_id,
    )


# ---------------------------------------------------------------------------
# RANSAC plane fitting
# ---------------------------------------------------------------------------

def _fit_plane_from_3_points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
                              ) -> Optional[Tuple[np.ndarray, float]]:
    """
    Fit a plane through 3 points. Returns (unit_normal, d) such that
    normal . point + d = 0 for points on the plane, or None if the
    points are collinear/degenerate.
    """
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)

    if norm_len < 1e-8:
        return None  # collinear points, no valid plane

    normal = normal / norm_len
    d = -np.dot(normal, p1)
    return normal, d


def _point_to_plane_distance(point: np.ndarray, normal: np.ndarray, d: float) -> float:
    """Perpendicular distance from a point to a plane defined by normal/d."""
    return float(abs(np.dot(normal, point) + d))


def _ransac_plane_fit(
    points       : List[np.ndarray],
    iterations   : int,
    threshold    : float,
    rng          : Optional[np.random.Generator] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], List[int]]:
    """
    Simple RANSAC plane fitting.

    Parameters
    ----------
    points     : list of 3D points (np.ndarray, shape (3,))
    iterations : number of random 3-point samples to try
    threshold  : inlier distance threshold (meters)
    rng        : optional numpy random generator for reproducibility

    Returns
    -------
    (best_normal, best_d, best_inlier_indices) — normal/d are None if
    no valid plane could be fit (e.g. too few points)
    """
    n = len(points)
    if n < 3:
        return None, None, []

    rng = rng or np.random.default_rng()

    best_inliers : List[int] = []
    best_normal  : Optional[np.ndarray] = None
    best_d       : Optional[float] = None

    for _ in range(iterations):
        idx_sample = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = (points[i] for i in idx_sample)

        fit = _fit_plane_from_3_points(p1, p2, p3)
        if fit is None:
            continue
        normal, d = fit

        inliers = [
            i for i, p in enumerate(points)
            if _point_to_plane_distance(p, normal, d) <= threshold
        ]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal  = normal
            best_d       = d

    return best_normal, best_d, best_inliers


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def filter_spatial_consistency(
    points : List[SpatialPoint],
    consts : SpatialConstants = SpatialConstants(),
    rng    : Optional[np.random.Generator] = None,
) -> SpatialFilterResult:
    """
    Run RANSAC plane fitting across a set of SpatialPoints from one scan
    to determine spatial consistency (i.e. do they describe a real surface
    or a chaotic smoke/dust cluster).

    Parameters
    ----------
    points : list of SpatialPoint, typically one per pulse in the scan
    consts : SpatialConstants (override for testing/tuning)
    rng    : optional numpy random generator for reproducibility

    Returns
    -------
    SpatialFilterResult
    """
    # Filter to candidate points worth fitting (decent surface evidence)
    candidates = [
        (i, p) for i, p in enumerate(points)
        if p.p_surface >= consts.MIN_P_SURFACE_FOR_FIT
    ]

    if len(candidates) < consts.MIN_POINTS_FOR_PLANE:
        return SpatialFilterResult(
            spatial_consistency  = 0.0,
            plane_found          = False,
            plane_normal         = None,
            inlier_indices        = [],
            outlier_indices       = [i for i, _ in candidates],
            num_candidate_points  = len(candidates),
        )

    candidate_indices = [i for i, _ in candidates]
    candidate_arrays  = [p.as_array() for _, p in candidates]

    normal, d, inlier_local_idx = _ransac_plane_fit(
        candidate_arrays,
        iterations=consts.RANSAC_ITERATIONS,
        threshold=consts.RANSAC_INLIER_THRESHOLD,
        rng=rng,
    )

    if normal is None:
        return SpatialFilterResult(
            spatial_consistency  = 0.0,
            plane_found          = False,
            plane_normal         = None,
            inlier_indices        = [],
            outlier_indices       = candidate_indices,
            num_candidate_points  = len(candidates),
        )

    inlier_fraction = len(inlier_local_idx) / len(candidates)
    plane_found = inlier_fraction >= consts.MIN_INLIER_FRACTION

    # Map local indices back to original point-list indices
    inlier_global  = [candidate_indices[i] for i in inlier_local_idx]
    outlier_global = [i for i in candidate_indices if i not in inlier_global]

    return SpatialFilterResult(
        spatial_consistency  = inlier_fraction,
        plane_found          = plane_found,
        plane_normal         = normal,
        inlier_indices        = inlier_global,
        outlier_indices       = outlier_global,
        num_candidate_points  = len(candidates),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pulse_classifier import classify_pulse
    from echo_physics import EchoReturn, extract_pulse_physics

    print("=" * 78)
    print("spatial_filter.py — Self-Test")
    print("=" * 78)

    rng = np.random.default_rng(42)

    def make_wall_pulse(azimuth, elevation, base_distance=6.0, noise=0.02):
        """A clean flat wall at ~6m, slight per-beam noise."""
        d = base_distance + rng.normal(0, noise)
        echoes = [EchoReturn(distance=d, intensity=0.4, echo_index=0, total_echoes=1)]
        return LidarPulse(echoes=echoes, beam_azimuth=azimuth, beam_elevation=elevation, timestamp=0.0)

    def make_smoke_pulse(azimuth, elevation):
        """Chaotic smoke return — random short distance, weak/variable intensity."""
        d = rng.uniform(1.0, 3.0)
        intensity = rng.uniform(0.05, 0.15)
        echoes = [EchoReturn(distance=d, intensity=intensity, echo_index=0, total_echoes=1)]
        return LidarPulse(echoes=echoes, beam_azimuth=azimuth, beam_elevation=elevation, timestamp=0.0)

    # Scenario A: clean wall scan — beams spread across BOTH azimuth and
    # elevation, all hitting the same flat wall (genuine 3D plane test)
    wall_points = []
    az_grid = np.linspace(-0.04, 0.04, 10)
    el_grid = np.linspace(-0.02, 0.02, 4)
    for az in az_grid:
        for el in el_grid:
            pulse = make_wall_pulse(az, el)
            features = extract_pulse_physics(pulse)
            result   = classify_pulse(pulse, features)
            pt = pulse_to_point(pulse, result)
            if pt:
                wall_points.append(pt)

    res_wall = filter_spatial_consistency(wall_points)
    print(f"\n[Scenario A: Clean flat wall, {len(wall_points)} beams, az+el spread]")
    print(f"  spatial_consistency : {round(res_wall.spatial_consistency, 3)}")
    print(f"  plane_found         : {res_wall.plane_found}")
    print(f"  num_candidates      : {res_wall.num_candidate_points}")
    print(f"  inliers/outliers    : {len(res_wall.inlier_indices)}/{len(res_wall.outlier_indices)}")

    # Scenario B: smoke scan — beams spread across BOTH azimuth and
    # elevation, each hitting an independently-random smoke return.
    # With elevation varying too, a point that randomly looks "clean"
    # still won't lie on a shared plane with the others.
    smoke_points = []
    for az in az_grid:
        for el in el_grid:
            pulse = make_smoke_pulse(az, el)
            features = extract_pulse_physics(pulse)
            result   = classify_pulse(pulse, features)
            pt = pulse_to_point(pulse, result)
            if pt:
                smoke_points.append(pt)

    res_smoke = filter_spatial_consistency(smoke_points)
    print(f"\n[Scenario B: Chaotic smoke, 10 beams]")
    print(f"  spatial_consistency : {round(res_smoke.spatial_consistency, 3)}")
    print(f"  plane_found         : {res_smoke.plane_found}")
    print(f"  num_candidates      : {res_smoke.num_candidate_points}")
    print(f"  inliers/outliers    : {len(res_smoke.inlier_indices)}/{len(res_smoke.outlier_indices)}")

    # Scenario C: mixed — half wall, half smoke (e.g. smoke drifting across part of a wall)
    mixed_points = wall_points[:20] + smoke_points[:20]
    res_mixed = filter_spatial_consistency(mixed_points)
    print(f"\n[Scenario C: Mixed — 20 wall points + 20 smoke points]")
    print(f"  spatial_consistency : {round(res_mixed.spatial_consistency, 3)}")
    print(f"  plane_found         : {res_mixed.plane_found}")
    print(f"  num_candidates      : {res_mixed.num_candidate_points}")
    print(f"  inliers/outliers    : {len(res_mixed.inlier_indices)}/{len(res_mixed.outlier_indices)}")

    print("\n" + "=" * 78)
    print("Expected: A high (~0.8-1.0), B low (~0.0-0.3), C in between.")
    print("=" * 78)