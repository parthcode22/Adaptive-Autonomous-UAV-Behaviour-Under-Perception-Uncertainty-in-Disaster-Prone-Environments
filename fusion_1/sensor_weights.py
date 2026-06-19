from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .ups_vector import FusionThresholds

@dataclass
class VisualReliabilityInput:
    """
    Fields coming from your existing Gabor pipeline output.
    Maps directly to what CoherenceMapComputer already produces.
    """
    global_confidence      : float   # main confidence score (informativeness-based)
    informativeness        : float   # energy + edge + contrast composite
    tier                   : str     # 'SAFE' or 'DANGER'
    scale_coherence_small  : float = 0.5
    scale_coherence_normal : float = 0.5
    scale_coherence_large  : float = 0.5

    def __post_init__(self):
        self.global_confidence      = float(np.clip(self.global_confidence,      0.0, 1.0))
        self.informativeness        = float(np.clip(self.informativeness,        0.0, 1.0))
        self.scale_coherence_small  = float(np.clip(self.scale_coherence_small,  0.0, 1.0))
        self.scale_coherence_normal = float(np.clip(self.scale_coherence_normal, 0.0, 1.0))
        self.scale_coherence_large  = float(np.clip(self.scale_coherence_large,  0.0, 1.0))


@dataclass
class LidarReliabilityInput:
    """
    Fields coming from the LiDAR multi-echo pipeline.
    Will be populated once lidar/ module is built.
    """
    p_surface            : float   # fraction of returns = real surfaces
    p_smoke              : float   # fraction of returns = smoke particles
    p_unknown            : float   # fraction of returns = ambiguous
    spatial_consistency  : float   # RANSAC planar fit quality (0-1)
    valid_echo_fraction  : float   # fraction of pulses that returned anything
    beta_estimate        : float = 0.0  # atmospheric extinction coefficient

    def __post_init__(self):
        self.p_surface           = float(np.clip(self.p_surface,           0.0, 1.0))
        self.p_smoke             = float(np.clip(self.p_smoke,             0.0, 1.0))
        self.p_unknown           = float(np.clip(self.p_unknown,           0.0, 1.0))
        self.spatial_consistency = float(np.clip(self.spatial_consistency, 0.0, 1.0))
        self.valid_echo_fraction = float(np.clip(self.valid_echo_fraction, 0.0, 1.0))
        self.beta_estimate= float(max(0.0, self.beta_estimate))

@dataclass
class SensorWeightResult:
    """
    Output of SensorWeightComputer.
    w_visual + w_lidar always = 1.0
    """
    w_visual      : float
    w_lidar       : float
    r_visual      : float   # raw reliability before normalisation
    r_lidar       : float   # raw reliability before normalisation
    weight_source : str     # diagnostic: VISUAL_DOMINANT / LIDAR_DOMINANT / BALANCED / BOTH_DEGRADED

    def __post_init__(self):
        total = self.w_visual + self.w_lidar
        if total > 1e-9:
            self.w_visual /= total
            self.w_lidar  /= total
        else:
            # both completely dead — equal floor weights
            self.w_visual = FusionThresholds.WEIGHT_FLOOR
            self.w_lidar  = FusionThresholds.WEIGHT_FLOOR


class SensorWeightComputer:

    def __init__(
        self,
        # visual sub-weights (must conceptually sum to 1.0)
        visual_w_confidence      : float = 0.50,
        visual_w_informativeness : float = 0.35,
        visual_w_scale_coherence : float = 0.15,

        # lidar sub-weights (must conceptually sum to 1.0)
        lidar_w_surface          : float = 0.35,
        lidar_w_smoke_penalty    : float = 0.25,
        lidar_w_unknown_penalty  : float = 0.20,
        lidar_w_spatial          : float = 0.15,
        lidar_w_echo_fraction    : float = 0.05,

        # sigmoid sharpening (prevents sluggish linear transitions)
        sigmoid_steepness        : float = 8.0,
        sigmoid_midpoint         : float = 0.5,

        # β extinction penalty
        beta_penalty_threshold   : float = 0.3,
        beta_penalty_scale       : float = 0.4,

        weight_floor             : float = FusionThresholds.WEIGHT_FLOOR,
    ):
        self.visual_w_confidence      = visual_w_confidence
        self.visual_w_informativeness = visual_w_informativeness
        self.visual_w_scale_coherence = visual_w_scale_coherence

        self.lidar_w_surface          = lidar_w_surface
        self.lidar_w_smoke_penalty    = lidar_w_smoke_penalty
        self.lidar_w_unknown_penalty  = lidar_w_unknown_penalty
        self.lidar_w_spatial          = lidar_w_spatial
        self.lidar_w_echo_fraction    = lidar_w_echo_fraction

        self.sigmoid_steepness        = sigmoid_steepness
        self.sigmoid_midpoint         = sigmoid_midpoint
        self.beta_penalty_threshold   = beta_penalty_threshold
        self.beta_penalty_scale       = beta_penalty_scale
        self.weight_floor             = weight_floor

    def compute(
        self,
        visual : VisualReliabilityInput,
        lidar  : LidarReliabilityInput,
    ) -> SensorWeightResult:

        r_visual = self._visual_reliability(visual)
        r_lidar  = self._lidar_reliability(lidar)

        # sigmoid sharpening — crisp transitions, not sluggish ramps
        r_v_sharp = self._sigmoid(r_visual)
        r_l_sharp = self._sigmoid(r_lidar)

        # apply floor — no sensor fully drops out
        r_v_final = max(r_v_sharp, self.weight_floor)
        r_l_final = max(r_l_sharp, self.weight_floor)

        source = self._source_label(r_v_final, r_l_final)

        return SensorWeightResult(
            w_visual      = r_v_final,
            w_lidar       = r_l_final,
            r_visual      = r_visual,
            r_lidar       = r_lidar,
            weight_source = source,
        )

    def _visual_reliability(self, v: VisualReliabilityInput) -> float:
        mean_scale = np.mean([
            v.scale_coherence_small,
            v.scale_coherence_normal,
            v.scale_coherence_large,
        ])

        raw = (
            self.visual_w_confidence      * v.global_confidence +
            self.visual_w_informativeness * v.informativeness   +
            self.visual_w_scale_coherence * mean_scale
        )

        # hard tier penalty
        if v.tier == 'DANGER':
            raw *= 0.60

        return float(np.clip(raw, 0.0, 1.0))

    def _lidar_reliability(self, l: LidarReliabilityInput) -> float:
        raw = (
            self.lidar_w_surface         * l.p_surface              +
            self.lidar_w_smoke_penalty   * (1.0 - l.p_smoke)        +
            self.lidar_w_unknown_penalty * (1.0 - l.p_unknown)       +
            self.lidar_w_spatial         * l.spatial_consistency      +
            self.lidar_w_echo_fraction   * l.valid_echo_fraction
        )

        # β extinction penalty — high β means heavy smoke absorption
        if l.beta_estimate > self.beta_penalty_threshold:
            excess  = l.beta_estimate - self.beta_penalty_threshold
            penalty = min(self.beta_penalty_scale * excess, 0.4)
            raw    -= penalty

        return float(np.clip(raw, 0.0, 1.0))
    def _sigmoid(self, x: float) -> float:
        """
        Maps [0,1] → [0,1] with steep transition near midpoint.
        Prevents weight changes from being too gradual.
        """
        k = self.sigmoid_steepness
        m = self.sigmoid_midpoint
        sig     = 1.0 / (1.0 + np.exp(-k * (x - m)))
        sig_min = 1.0 / (1.0 + np.exp(-k * (0.0 - m)))
        sig_max = 1.0 / (1.0 + np.exp(-k * (1.0 - m)))
        return float(np.clip((sig - sig_min) / (sig_max - sig_min + 1e-9), 0.0, 1.0))

    def _source_label(self, r_v: float, r_l: float) -> str:
        total = r_v + r_l
        if total < 2 * self.weight_floor + 0.01:
            return "BOTH_DEGRADED"
        w_v = r_v / total
        w_l = r_l / total
        thresh = FusionThresholds.SENSOR_DOMINANT_MIN
        if w_v >= thresh: return "VISUAL_DOMINANT"
        if w_l >= thresh: return "LIDAR_DOMINANT"
        return "BALANCED_FUSION"
    
