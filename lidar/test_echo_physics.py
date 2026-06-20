"""
test_echo_physics.py — Standalone Validation Script
=====================================================
SURE-UAV LiDAR Smoke Filtering Module

Run this independently to sanity-check echo_physics.py against synthetic
ground-truth scenarios. Edit the SCENARIOS list freely to try different
distance/intensity combinations and see how beta/ΔR/intensity_ratio respond.

Usage:
    python test_echo_physics.py

Edit and re-run as many times as needed — this file does not affect
echo_physics.py itself.
"""

from echo_physics import (
    EchoReturn,
    LidarPulse,
    PhysicsConstants,
    range_corrected_intensity,
    estimate_beta,
    extract_pulse_physics,
)


# ---------------------------------------------------------------------------
# Helper to build a pulse from a simple list of (distance, intensity) tuples
# ---------------------------------------------------------------------------

def make_pulse(echo_specs, label):
    """
    echo_specs : list of (distance_m, intensity_0to1) tuples
    label      : human-readable description of what this pulse represents
    """
    echoes = [
        EchoReturn(
            distance=d, intensity=i,
            echo_index=idx, total_echoes=len(echo_specs)
        )
        for idx, (d, i) in enumerate(echo_specs)
    ]
    pulse = LidarPulse(
        echoes=echoes,
        beam_azimuth=0.0, beam_elevation=0.0, timestamp=0.0,
    )
    return pulse, label


# ---------------------------------------------------------------------------
# EDIT THIS LIST — add/remove/change scenarios freely
# ---------------------------------------------------------------------------

SCENARIOS = [

    # 1. Clean wall, close range — single strong echo
    make_pulse(
        [(2.0, 0.45)],
        "1. Clean wall, close range (2m, strong return)"
    ),

    # 2. Clean wall, far range — single echo, weaker due to distance alone
    #    Chosen so range_corrected_intensity matches scenario 1
    #    (intensity * R^2 should be similar) -> beta should still be ~0
    make_pulse(
        [(10.0, 0.018)],
        "2. Clean wall, far range (10m, range-falloff-only weak)"
    ),

    # 3. Smoke-then-wall — weak close echo (smoke), strong far echo (real wall)
    make_pulse(
        [(2.0, 0.05), (9.0, 0.50)],
        "3. Smoke-then-wall (weak near, strong far)"
    ),

    # 4. Dust cloud — multiple weak echoes, decaying intensity, small ΔR
    #    NOTE: original values here did NOT trigger beta detection.
    #    Try lowering intensities further or adjusting RHO_REF if you
    #    want this scenario to register nonzero beta.
    make_pulse(
    [(3.0, 0.03), (3.6, 0.018), (4.1, 0.01)],
    "4. Dust cloud (decaying, tightly spaced)"
    ),
    # 5. Total absorption — zero echoes
    make_pulse(
        [],
        "5. Total absorption (dense smoke, no return at all)"
    ),

    # 6. Low-reflectivity but REAL surface — single weak echo, moderate range
    #    Should NOT look like smoke — no second stronger echo exists.
    make_pulse(
        [(5.0, 0.04)],
        "6. Low-reflectivity real surface (dark wall, 5m)"
    ),

    # --- Add your own scenarios below ---
    # make_pulse(
    #     [(distance, intensity), (distance, intensity), ...],
    #     "Description of what this represents"
    # ),

]


# ---------------------------------------------------------------------------
# Run all scenarios and print results
# ---------------------------------------------------------------------------

def run_tests():
    print("=" * 78)
    print("echo_physics.py — Standalone Test")
    print(f"RHO_REF = {PhysicsConstants.RHO_REF}   "
          f"MIN_RANGE_FOR_BETA = {PhysicsConstants.MIN_RANGE_FOR_BETA}   "
          f"MAX_BETA = {PhysicsConstants.MAX_BETA}")
    print("=" * 78)

    for pulse, label in SCENARIOS:
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

        # Quick per-scenario sanity flags (edit thresholds as you like)
        if features.has_return and features.echo_count == 1:
            flag = "OK — single clean echo" if features.beta_estimate < 0.05 \
                   else "FLAGGED — single echo but beta > 0 (extinction detected)"
            print(f"  >> {flag}")

    print("\n" + "=" * 78)
    print("Done. Adjust SCENARIOS or PhysicsConstants in echo_physics.py and re-run.")
    print("=" * 78)


if __name__ == "__main__":
    run_tests()