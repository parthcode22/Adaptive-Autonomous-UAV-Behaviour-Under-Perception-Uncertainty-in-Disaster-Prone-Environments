from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

from echo_physics import LidarPulse, PulsePhysicsFeatures, PhysicsConstants

class ClassifierConstants:
    """
    Calibration constants for pulse-level smoke/surface evidence scoring.
    """
    # --- p_surface scoring ---
    # Beta below this on a given echo = strong surface evidence
    SURFACE_BETA_CLEAN_MAX     : float = 0.05
    # Range-corrected intensity above this = confident real reflectivity
    SURFACE_INTENSITY_STRONG   : float = 1.2

    # --- p_smoke scoring ---
    # Beta above this on any echo = meaningful extinction evidence
    SMOKE_BETA_DETECT_MIN      : float = 0.05
    # Beta above this = strong/unambiguous smoke evidence
    SMOKE_BETA_STRONG          : float = 0.3
    # Range separation below this (with multiple echoes) suggests
    # a diffuse scattering medium (dust) rather than a single hard surface
    SMOKE_TIGHT_SPACING_MAX    : float = 1.0   # meters
    # Intensity ratio below this between consecutive echoes suggests
    # decay through a scattering medium
    SMOKE_DECAY_RATIO_MAX      : float = 0.85

    # --- p_unknown scoring ---
    # If no echo's range-corrected intensity exceeds this, the pulse
    # carries little usable signal in either direction
    UNKNOWN_WEAK_SIGNAL_MAX    : float = 0.15

@dataclass
class PulseClassification:
    """
    Independent evidence scores for a single pulse.

    Attributes
    ----------
    p_surface       : confidence a usable surface return exists (0-1)
    p_smoke         : confidence smoke/dust interference exists (0-1)
    p_unknown       : confidence the pulse is unexplained by either (0-1)
    best_surface_echo_index : index of the echo driving p_surface
                              (-1 if no echoes / has_return is False)
    evidence_notes  : short diagnostic string, e.g. "smoke_then_surface",
                      "dust_pattern", "clean_single_echo", "total_absorption"
    """
    p_surface: float
    p_smoke: float
    p_unknown: float
    best_surface_echo_index: int
    evidence_notes: str

    def __post_init__(self):
        self.p_surface = float(np.clip(self.p_surface, 0.0, 1.0))
        self.p_smoke = float(np.clip(self.p_smoke, 0.0, 1.0))
        self.p_unknown = float(np.clip(self.p_unknown, 0.0, 1.0))


def _echo_surface_evidence(
    beta: float,
    range_corrected_intensity: float
) -> float:

    c = ClassifierConstants

    # Cleanliness score: 1.0 at beta=0, decaying to 0 as beta rises
    cleanliness = float(
        np.clip(
            1.0 - (beta / c.SURFACE_BETA_CLEAN_MAX),
            0.0,
            1.0
        )
    ) if beta <= c.SURFACE_BETA_CLEAN_MAX else 0.0

    # Smoother decay above the clean threshold rather than a hard cliff —
    # use an exponential falloff for beta beyond the clean max
    if beta > c.SURFACE_BETA_CLEAN_MAX:
        cleanliness = float(
            np.exp(
                -(beta - c.SURFACE_BETA_CLEAN_MAX) * 15.0
            )
        )

    # Strength score: scaled against the "strong" reference intensity
    strength = float(
        np.clip(
            range_corrected_intensity / c.SURFACE_INTENSITY_STRONG,
            0.0,
            1.0
        )
    )

    # Both must contribute — geometric mean penalizes a 0 in either factor
    # harder than an arithmetic mean would
    evidence = float(np.sqrt(cleanliness * strength))

    return evidence


def _compute_p_surface(pulse: LidarPulse, features: PulsePhysicsFeatures) -> tuple[float, int]:
   
    if not features.has_return:
        return 0.0, -1

    echoes = pulse.echoes
    best_score = -1.0
    best_index = -1

    for i, echo in enumerate(echoes):
        beta = features.beta_per_echo[i]
        rc_intensity = echo.intensity * (max(echo.distance, PhysicsConstants.MIN_RANGE_FOR_BETA) ** 2)

        score = _echo_surface_evidence(beta, rc_intensity)

        if score > best_score:
            best_score = score
            best_index = i

    return float(np.clip(best_score, 0.0, 1.0)), best_index


def _echo_smoke_evidence(beta: float) -> float:
    """
    Score a single echo on how strongly it shows evidence of atmospheric
    scattering (smoke/dust), based purely on its extinction coefficient.

    Parameters
    ----------
    beta : estimated extinction coefficient for this echo

    Returns
    -------
    float (0-1) — smoke evidence score for this single echo
    """
    c = ClassifierConstants

    if beta < c.SMOKE_BETA_DETECT_MIN:
        return 0.0

    # Linear ramp from DETECT_MIN to STRONG, saturating at 1.0 beyond STRONG
    evidence = (beta - c.SMOKE_BETA_DETECT_MIN) / (c.SMOKE_BETA_STRONG - c.SMOKE_BETA_DETECT_MIN)
    return float(np.clip(evidence, 0.0, 1.0))


def _dust_pattern_bonus(features: PulsePhysicsFeatures) -> float:
    """
    Detect the 'dust cloud' signature: multiple echoes, tightly spaced,
    with gradually decaying intensity. This pattern itself is evidence
    of smoke/dust even if individual beta values are moderate.

    Parameters
    ----------
    features : PulsePhysicsFeatures from echo_physics.py

    Returns
    -------
    float (0-1) — bonus evidence score from the multi-echo pattern alone
    """
    c = ClassifierConstants

    if features.echo_count < 2:
        return 0.0

    # Check spacing: are echoes tightly packed?
    tight_spacing = all(
        sep <= c.SMOKE_TIGHT_SPACING_MAX for sep in features.range_separations
    )

    # Check decay: are intensities gradually decreasing (ratio < 1)?
    decaying = all(
        ratio <= c.SMOKE_DECAY_RATIO_MAX for ratio in features.intensity_ratios
    )

    if tight_spacing and decaying:
        # Scale bonus by how many echoes confirm the pattern
        return float(np.clip(0.3 + 0.15 * features.echo_count, 0.0, 1.0))

    return 0.0


def _compute_p_smoke(features: PulsePhysicsFeatures) -> float:
    """
    Compute p_smoke by combining the strongest single-echo evidence
    with the multi-echo dust pattern bonus.

    Parameters
    ----------
    features : PulsePhysicsFeatures from echo_physics.py

    Returns
    -------
    float (0-1) — p_smoke for this pulse
    """
    if not features.has_return:
        # Total absorption is itself a near-maximal smoke signal —
        # the laser energy was fully scattered/absorbed somewhere
        return 1.0

    per_echo_scores = [_echo_smoke_evidence(b) for b in features.beta_per_echo]
    max_echo_evidence = max(per_echo_scores) if per_echo_scores else 0.0

    pattern_bonus = _dust_pattern_bonus(features)

    # Combine: take the stronger of the two signals, then let the other
    # nudge it up slightly rather than double-counting fully
    combined = max(max_echo_evidence, pattern_bonus) + 0.2 * min(max_echo_evidence, pattern_bonus)

    return float(np.clip(combined, 0.0, 1.0))


def _compute_p_unknown(features: PulsePhysicsFeatures, p_surface: float, p_smoke: float) -> float:
    """
    Compute p_unknown: how much of this pulse's behavior is unexplained
    by either surface or smoke evidence.

    High when:
    - No return at all but we already capped p_smoke at 1.0 for that case
      (handled separately below — total absorption is NOT "unknown",
      it's a strong smoke signal, so p_unknown should be LOW there)
    - Weak signal everywhere AND neither p_surface nor p_smoke scored high

    Parameters
    ----------
    features  : PulsePhysicsFeatures from echo_physics.py
    p_surface : already-computed p_surface for this pulse
    p_smoke   : already-computed p_smoke for this pulse

    Returns
    -------
    float (0-1) — p_unknown for this pulse
    """
    c = ClassifierConstants

    if not features.has_return:
        # Total absorption is well-explained by p_smoke=1.0 already
        return 0.0

    # If both surface and smoke evidence are weak, and the signal itself
    # is weak, this pulse is genuinely ambiguous
    max_rc_intensity = max(features.range_corrected_first, features.range_corrected_last)
    weak_signal = max_rc_intensity < c.UNKNOWN_WEAK_SIGNAL_MAX

    neither_confident = (p_surface < 0.3) and (p_smoke < 0.3)

    if weak_signal and neither_confident:
        return float(np.clip(1.0 - max(p_surface, p_smoke), 0.4, 1.0))

    return float(np.clip(1.0 - max(p_surface, p_smoke) * 1.5, 0.0, 0.4))


def classify_pulse(pulse: LidarPulse, features: PulsePhysicsFeatures) -> PulseClassification:
    """
    Main entry point: classify a single pulse into independent
    p_surface / p_smoke / p_unknown evidence scores.

    Parameters
    ----------
    pulse    : original LidarPulse
    features : PulsePhysicsFeatures from echo_physics.extract_pulse_physics()

    Returns
    -------
    PulseClassification
    """
    p_surface, best_idx = _compute_p_surface(pulse, features)
    p_smoke             = _compute_p_smoke(features)
    p_unknown           = _compute_p_unknown(features, p_surface, p_smoke)

    # Diagnostic label
    if not features.has_return:
        notes = "total_absorption"
    elif p_surface > 0.6 and p_smoke > 0.3:
        notes = "smoke_then_surface"
    elif p_smoke > 0.5 and features.echo_count >= 2:
        notes = "dust_pattern"
    elif p_surface > 0.6:
        notes = "clean_surface"
    elif p_unknown > 0.5:
        notes = "ambiguous"
    else:
        notes = "mixed_evidence"

    return PulseClassification(
        p_surface=p_surface,
        p_smoke=p_smoke,
        p_unknown=p_unknown,
        best_surface_echo_index=best_idx,
        evidence_notes=notes,
    )