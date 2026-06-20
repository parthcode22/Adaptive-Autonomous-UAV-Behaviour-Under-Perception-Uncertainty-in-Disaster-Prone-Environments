"""
temporal_filter.py — Temporal Persistence Filter
==================================================
SURE-UAV LiDAR Smoke Filtering Module

Tracks surface points across multiple scans over time. A real wall stays
in roughly the same place scan after scan. Smoke drifts, flickers, or
disappears entirely between scans.

This is the layer that catches what spatial_filter.py CAN'T see —
a single scan has no way to know if a point is stable over time.
Even a spatially "planar-looking" cluster of smoke (e.g. a dust sheet
caught mid-drift) will fail to persist across consecutive scans.

Approach: for each point in the current scan, search for a nearby point
in each of the last N scans (within a distance tolerance). The fraction
of recent scans where a match was found is the persistence_score.
"""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional
import numpy as np

from spatial_filter import SpatialPoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TemporalConstants:
    """Calibration constants for temporal persistence tracking."""

    # How many past scans to retain in history
    HISTORY_LENGTH           : int   = 5

    # Maximum distance (meters) for a point in scan N-1 to "match"
    # a point in scan N as the same physical feature
    MATCH_DISTANCE_THRESHOLD : float = 0.20

    # Minimum number of past scans required before persistence_score
    # is considered meaningful (early scans default to neutral score)
    MIN_HISTORY_FOR_SCORE    : int   = 2

    # Persistence score below this flags a point as likely transient/smoke
    PERSISTENCE_LOW          : float = 0.3

    # Persistence score above this flags a point as likely stable/structure
    PERSISTENCE_HIGH         : float = 0.7


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TemporalPointResult:
    """
    Persistence result for a single point in the current scan.

    Attributes
    ----------
    point             : the SpatialPoint being scored
    persistence_score : float (0-1) — fraction of recent scans with a match
    matched_scan_count: how many of the available history scans matched
    history_available : how many past scans were actually available
                         (may be less than HISTORY_LENGTH early on)
    """
    point              : SpatialPoint
    persistence_score  : float
    matched_scan_count : int
    history_available  : int

    def __post_init__(self):
        self.persistence_score = float(np.clip(self.persistence_score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Temporal tracker (stateful — call update() once per scan)
# ---------------------------------------------------------------------------

class TemporalTracker:
    """
    Maintains a rolling history of past scans and scores each new scan's
    points for temporal persistence.

    Usage
    -----
        tracker = TemporalTracker()

        for scan_points in scan_sequence:
            results = tracker.update(scan_points)
            for r in results:
                print(r.persistence_score)
    """

    def __init__(self, consts: TemporalConstants = TemporalConstants()):
        self.consts  = consts
        self._history: Deque[List[SpatialPoint]] = deque(maxlen=consts.HISTORY_LENGTH)

    def update(self, current_scan: List[SpatialPoint]) -> List[TemporalPointResult]:
        """
        Score the current scan's points against scan history, then push
        the current scan into history for future calls.

        Parameters
        ----------
        current_scan : list of SpatialPoint from the current scan

        Returns
        -------
        List[TemporalPointResult], one per point in current_scan, in the
        same order as the input list.
        """
        history_available = len(self._history)
        results: List[TemporalPointResult] = []

        for point in current_scan:
            if history_available < self.consts.MIN_HISTORY_FOR_SCORE:
                # Not enough history yet — neutral score, not penalized
                results.append(TemporalPointResult(
                    point=point,
                    persistence_score=0.5,
                    matched_scan_count=0,
                    history_available=history_available,
                ))
                continue

            matched_count = 0
            for past_scan in self._history:
                if self._has_nearby_match(point, past_scan):
                    matched_count += 1

            score = matched_count / history_available

            results.append(TemporalPointResult(
                point=point,
                persistence_score=score,
                matched_scan_count=matched_count,
                history_available=history_available,
            ))

        # Push current scan into history AFTER scoring (so a point doesn't
        # match against itself)
        self._history.append(current_scan)

        return results

    def reset(self):
        """Clear all history. Call between missions or test runs."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Internal matching
    # ------------------------------------------------------------------

    def _has_nearby_match(
        self,
        point     : SpatialPoint,
        past_scan : List[SpatialPoint],
    ) -> bool:
        """
        Check if any point in past_scan lies within MATCH_DISTANCE_THRESHOLD
        of the given point.
        """
        if not past_scan:
            return False

        p = point.as_array()
        threshold = self.consts.MATCH_DISTANCE_THRESHOLD

        for past_point in past_scan:
            dist = float(np.linalg.norm(p - past_point.as_array()))
            if dist <= threshold:
                return True

        return False


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 78)
    print("temporal_filter.py — Self-Test")
    print("=" * 78)

    rng = np.random.default_rng(7)

    def make_point(x, y, z, p_surface=0.8, p_smoke=0.0):
        return SpatialPoint(x=x, y=y, z=z, p_surface=p_surface, p_smoke=p_smoke)

    # --- Scenario A: stable wall point across 6 scans (tiny sensor noise) ---
    tracker_a = TemporalTracker()
    base_x, base_y, base_z = 5.0, 0.5, 0.2
    print("\n[Scenario A: Stable wall point, 6 scans, mm-scale noise]")
    for scan_idx in range(6):
        noisy_point = make_point(
            base_x + rng.normal(0, 0.01),
            base_y + rng.normal(0, 0.01),
            base_z + rng.normal(0, 0.01),
        )
        results = tracker_a.update([noisy_point])
        r = results[0]
        print(f"  scan {scan_idx}: persistence_score={round(r.persistence_score,3)} "
              f"matched={r.matched_scan_count}/{r.history_available}")

    # --- Scenario B: smoke point — jumps to a new random position every scan ---
    tracker_b = TemporalTracker()
    print("\n[Scenario B: Drifting smoke point, 6 scans, random position each time]")
    for scan_idx in range(6):
        drifting_point = make_point(
            rng.uniform(1.0, 5.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-0.5, 0.5),
        )
        results = tracker_b.update([drifting_point])
        r = results[0]
        print(f"  scan {scan_idx}: persistence_score={round(r.persistence_score,3)} "
              f"matched={r.matched_scan_count}/{r.history_available}")

    # --- Scenario C: mixed scan — one stable wall point + one drifting smoke point ---
    tracker_c = TemporalTracker()
    print("\n[Scenario C: Mixed scan, 6 scans, 1 stable + 1 drifting point]")
    for scan_idx in range(6):
        stable_point = make_point(
            base_x + rng.normal(0, 0.01),
            base_y + rng.normal(0, 0.01),
            base_z + rng.normal(0, 0.01),
        )
        drifting_point = make_point(
            rng.uniform(1.0, 5.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-0.5, 0.5),
        )
        results = tracker_c.update([stable_point, drifting_point])
        stable_r, drift_r = results
        print(f"  scan {scan_idx}: stable={round(stable_r.persistence_score,3)}  "
              f"drifting={round(drift_r.persistence_score,3)}")

    print("\n" + "=" * 78)
    print("Expected: A converges HIGH (~1.0), B stays LOW (~0.0),")
    print("C shows stable point HIGH and drifting point LOW by the final scans.")
    print("=" * 78)