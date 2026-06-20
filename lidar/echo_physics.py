"""
echo_physics.py — Physical Foundation Layer for Multi-Echo LiDAR
==================================================================
SURE-UAV LiDAR Smoke Filtering Module

Defines the core data structures (EchoReturn, LidarPulse) and implements
the LiDAR range equation inversion to estimate the atmospheric extinction
coefficient (beta) — the physical basis for distinguishing smoke
attenuation from genuine low-reflectivity surfaces or range falloff.

This file does NOT classify anything. It only produces physically
grounded features that pulse_classifier.py will consume.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Block 2 — EchoReturn
# ---------------------------------------------------------------------------

@dataclass
class EchoReturn:
    """
    A single LiDAR return from one laser pulse.

    Attributes
    ----------
    distance     : range to this return, in meters
    intensity    : raw return intensity, normalized to [0, 1]
    echo_index   : position within the pulse (0 = first echo received)
    total_echoes : total number of echoes detected for the parent pulse
    """
    distance     : float
    intensity    : float
    echo_index   : int
    total_echoes : int

    def __post_init__(self):
        self.distance  = float(max(0.0, self.distance))
        self.intensity = float(np.clip(self.intensity, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Block 3 — LidarPulse
# ---------------------------------------------------------------------------

@dataclass
class LidarPulse:
    """
    A single laser pulse and all echoes it produced.

    Attributes
    ----------
    echoes         : all EchoReturn objects from this pulse, ordered by echo_index
    beam_azimuth   : horizontal beam angle, radians (0 = forward, +ve = right)
    beam_elevation : vertical beam angle, radians (0 = level, +ve = up)
    timestamp      : time this pulse was fired/received, seconds
    pulse_id       : optional unique identifier, useful for debugging/tracking
    """
    echoes         : List[EchoReturn]
    beam_azimuth   : float
    beam_elevation : float
    timestamp      : float
    pulse_id       : Optional[int] = None

    def __post_init__(self):
        # keep echoes sorted by echo_index — every downstream layer assumes this
        self.echoes = sorted(self.echoes, key=lambda e: e.echo_index)

    @property
    def echo_count(self) -> int:
        """Number of echoes detected for this pulse."""
        return len(self.echoes)

    @property
    def first_echo(self) -> Optional[EchoReturn]:
        """The first (nearest) echo, or None if pulse has no returns."""
        return self.echoes[0] if self.echoes else None

    @property
    def last_echo(self) -> Optional[EchoReturn]:
        """The last (farthest) echo, or None if pulse has no returns."""
        return self.echoes[-1] if self.echoes else None

    @property
    def has_return(self) -> bool:
        """False if the laser energy was fully absorbed/scattered — no echo at all."""
        return self.echo_count > 0


# ---------------------------------------------------------------------------
# Block 4 — PhysicsConstants
# ---------------------------------------------------------------------------

class PhysicsConstants:
    """
    Calibration constants for the LiDAR range equation inversion.
    Centralised here — tune only in one place.
    """
    # Reference reflectivity for a "typical" surface in clear air.
    # Calibrated empirically — same role as Gabor's ENERGY_REF.
    RHO_REF: float = 0.5

    # Minimum range to avoid division-by-zero / unstable beta estimates
    # for extremely close returns.
    MIN_RANGE_FOR_BETA: float = 0.15   # meters

    # Maximum physically plausible beta — heavy smoke/fog ceilings out
    # around here; anything beyond is clipped (sensor noise, not signal).
    MAX_BETA: float = 2.0


# ---------------------------------------------------------------------------
# Block 5 — Core physics functions
# ---------------------------------------------------------------------------

def range_corrected_intensity(intensity: float, distance: float) -> float:
    """
    Remove the inverse-square range falloff from a raw intensity reading.

    Physically: I_received ∝ 1/R², so multiplying by R² gives us the
    intensity we'd expect to see if range falloff alone explained the signal.
    What's LEFT after this correction is attributable to reflectivity
    and/or atmospheric extinction.

    Parameters
    ----------
    intensity : raw return intensity, normalized [0, 1]
    distance  : range to this return, meters

    Returns
    -------
    float — range-corrected intensity (can exceed 1.0 for close, bright returns)
    """
    r = max(distance, PhysicsConstants.MIN_RANGE_FOR_BETA)
    return float(intensity * (r ** 2))


def estimate_beta(intensity: float, distance: float) -> float:
    """
    Estimate the atmospheric extinction coefficient (beta) for a single echo.

    Inverts the two-way LiDAR range equation:
        I_received = (rho_ref / R²) × exp(-2*beta*R)
    solved for beta:
        beta = -ln(I_received × R² / rho_ref) / (2R)

    A return at least as strong as the reference (clean air) produces
    beta <= 0, which we clip to 0.0 — no detectable smoke.
    A weaker-than-reference return produces beta > 0, scaling with
    how much weaker it is and how far away it was measured.

    Parameters
    ----------
    intensity : raw return intensity, normalized [0, 1]
    distance  : range to this return, meters

    Returns
    -------
    float — estimated beta, clipped to [0, PhysicsConstants.MAX_BETA]
    """
    r = max(distance, PhysicsConstants.MIN_RANGE_FOR_BETA)
    corrected = range_corrected_intensity(intensity, r)
    attenuation_ratio = corrected / PhysicsConstants.RHO_REF

    if attenuation_ratio >= 1.0:
        # return is as strong or stronger than reference — no excess attenuation
        return 0.0

    if attenuation_ratio <= 1e-6:
        # essentially no signal — cap at max beta rather than computing log(~0)
        return PhysicsConstants.MAX_BETA

    beta = -np.log(attenuation_ratio) / (2.0 * r)
    return float(np.clip(beta, 0.0, PhysicsConstants.MAX_BETA))


# ---------------------------------------------------------------------------
# Block 6 — PulsePhysicsFeatures
# ---------------------------------------------------------------------------

@dataclass
class PulsePhysicsFeatures:
    """
    Physically grounded features computed across all echoes in a single pulse.
    This is the output of echo_physics.py — consumed by pulse_classifier.py.

    Attributes
    ----------
    echo_count             : number of echoes in this pulse
    has_return             : False if pulse had zero echoes (full absorption)
    range_separations      : list of ΔR between consecutive echoes, meters
                              (empty if echo_count <= 1)
    intensity_ratios       : list of I_(n+1)/I_n between consecutive echoes
                              (empty if echo_count <= 1)
    beta_per_echo          : estimated extinction coefficient for each echo
    beta_estimate           : representative beta for the whole pulse
                              (currently: from the LAST echo — the most likely
                              true surface, but pulse_classifier.py will weigh
                              this against other evidence, not assume it)
    range_corrected_first  : range-corrected intensity of the first echo
    range_corrected_last   : range-corrected intensity of the last echo
    max_range_separation   : largest single ΔR in the pulse (0.0 if N<=1)
    mean_intensity_ratio    : average intensity ratio across echoes (1.0 if N<=1)
    """
    echo_count            : int
    has_return            : bool
    range_separations     : List[float]
    intensity_ratios      : List[float]
    beta_per_echo         : List[float]
    beta_estimate         : float
    range_corrected_first : float
    range_corrected_last  : float
    max_range_separation  : float
    mean_intensity_ratio   : float


# ---------------------------------------------------------------------------
# Block 7 — extract_pulse_physics()
# ---------------------------------------------------------------------------

def extract_pulse_physics(pulse: LidarPulse) -> PulsePhysicsFeatures:
    """
    Compute physically grounded features from a single LidarPulse.

    This is a PURE function of physics — it makes no smoke/surface
    classification decisions. It only describes what the echoes physically
    imply, for pulse_classifier.py to interpret next.

    Parameters
    ----------
    pulse : LidarPulse with echoes already sorted by echo_index (guaranteed
            by LidarPulse.__post_init__)

    Returns
    -------
    PulsePhysicsFeatures
    """
    if not pulse.has_return:
        # No echo at all — laser energy fully absorbed/scattered.
        # This is itself informative (handled by valid_echo_fraction upstream)
        # but there is no physics to compute here.
        return PulsePhysicsFeatures(
            echo_count            = 0,
            has_return            = False,
            range_separations     = [],
            intensity_ratios      = [],
            beta_per_echo         = [],
            beta_estimate         = PhysicsConstants.MAX_BETA,  # total absorption ≈ max extinction
            range_corrected_first = 0.0,
            range_corrected_last  = 0.0,
            max_range_separation  = 0.0,
            mean_intensity_ratio   = 1.0,
        )

    echoes = pulse.echoes
    n      = len(echoes)

    # Per-echo beta estimates
    beta_per_echo = [
        estimate_beta(e.intensity, e.distance) for e in echoes
    ]

    # Range-corrected intensities for first and last echo
    rc_first = range_corrected_intensity(echoes[0].intensity,  echoes[0].distance)
    rc_last  = range_corrected_intensity(echoes[-1].intensity, echoes[-1].distance)

    if n == 1:
        # Single echo — no inter-echo relationships exist
        return PulsePhysicsFeatures(
            echo_count            = 1,
            has_return            = True,
            range_separations     = [],
            intensity_ratios      = [],
            beta_per_echo         = beta_per_echo,
            beta_estimate         = beta_per_echo[0],
            range_corrected_first = rc_first,
            range_corrected_last  = rc_last,
            max_range_separation  = 0.0,
            mean_intensity_ratio   = 1.0,
        )

    # Multiple echoes — compute inter-echo features
    range_separations = [
        echoes[i + 1].distance - echoes[i].distance
        for i in range(n - 1)
    ]

    intensity_ratios = [
        echoes[i + 1].intensity / max(echoes[i].intensity, 1e-6)
        for i in range(n - 1)
    ]

    # Representative beta: weight toward the LAST echo (most likely true
    # surface per our design decision — but pulse_classifier.py decides
    # how much to actually trust this, not us)
    beta_estimate = max(beta_per_echo)

    return PulsePhysicsFeatures(
        echo_count            = n,
        has_return            = True,
        range_separations     = range_separations,
        intensity_ratios      = intensity_ratios,
        beta_per_echo         = beta_per_echo,
        beta_estimate         = beta_estimate,
        range_corrected_first = rc_first,
        range_corrected_last  = rc_last,
        max_range_separation  = float(max(range_separations)),
        mean_intensity_ratio   = float(np.mean(intensity_ratios)),
    )


# ---------------------------------------------------------------------------
# Self-Test — synthetic scenarios
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    def make_pulse(echo_specs, label):
        """
        echo_specs: list of (distance, intensity) tuples
        """
        echoes = [
            EchoReturn(
                distance=d, intensity=i,
                echo_index=idx, total_echoes=len(echo_specs)
            )
            for idx, (d, i) in enumerate(echo_specs)
        ]
        return LidarPulse(
            echoes=echoes,
            beam_azimuth=0.0, beam_elevation=0.0, timestamp=0.0,
        ), label

    scenarios = [
        # 1. Clean wall, close range — single strong echo
        make_pulse([(2.0, 0.45)], "1. Clean wall, close range (2m, strong return)"),

        # 2. Clean wall, far range — single echo, weaker due to distance alone
        #    Intensity chosen so range_corrected_intensity ≈ same as scenario 1
        #    i.e. intensity * R^2 should be similar -> beta should still be ~0
        make_pulse([(10.0, 0.018)], "2. Clean wall, far range (10m, range-falloff-only weak)"),

        # 3. Smoke-then-wall — weak close echo (smoke), strong far echo (real wall)
        make_pulse([(2.0, 0.05), (9.0, 0.50)], "3. Smoke-then-wall (weak near, strong far)"),

        # 4. Dust cloud — multiple weak echoes, decaying intensity, small ΔR
        make_pulse([(3.0, 0.20), (3.6, 0.12), (4.1, 0.07)], "4. Dust cloud (decaying, tightly spaced)"),

        # 5. Total absorption — zero echoes
        make_pulse([], "5. Total absorption (dense smoke, no return at all)"),

        # 6. Low-reflectivity but REAL surface — single weak echo, moderate range
        #    Should NOT look identical to "smoke" — no second stronger echo exists,
        #    and intensity is weak but range-corrected intensity is still
        #    plausible for a dark surface, not zero.
        make_pulse([(5.0, 0.04)], "6. Low-reflectivity real surface (dark wall, 5m)"),
    ]

    print("=" * 78)
    print("echo_physics.py — Self-Test")
    print("=" * 78)

    for pulse, label in scenarios:
        features = extract_pulse_physics(pulse)
        print(f"\n[{label}]")
        print(f"  echo_count            : {features.echo_count}")
        print(f"  has_return            : {features.has_return}")
        print(f"  beta_per_echo         : {[round(b, 3) for b in features.beta_per_echo]}")
        print(f"  beta_estimate         : {round(features.beta_estimate, 3)}")
        print(f"  range_separations     : {[round(r, 2) for r in features.range_separations]}")
        print(f"  intensity_ratios      : {[round(r, 2) for r in features.intensity_ratios]}")
        print(f"  range_corrected_first : {round(features.range_corrected_first, 3)}")
        print(f"  range_corrected_last  : {round(features.range_corrected_last, 3)}")
        print(f"  max_range_separation  : {round(features.max_range_separation, 2)}")
        print(f"  mean_intensity_ratio  : {round(features.mean_intensity_ratio, 2)}")

    print("\n" + "=" * 78)
    print("Self-test complete. Review beta/ΔR/intensity_ratio values above")
    print("against the physical scenario each represents.")
    print("=" * 78)