"""
test_pulse_classifier.py — Standalone Validation Script
==========================================================
SURE-UAV LiDAR Smoke Filtering Module

Run this independently to check p_surface / p_smoke / p_unknown
against the same 6 synthetic scenarios validated in echo_physics.py.

Usage:
    python test_pulse_classifier.py
"""

from echo_physics import EchoReturn, LidarPulse, extract_pulse_physics
from pulse_classifier import classify_pulse


def make_pulse(echo_specs, label):
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


SCENARIOS = [

    make_pulse(
        [(2.0, 0.45)],
        "1. Clean wall, close range (2m, strong return)"
    ),

    make_pulse(
        [(10.0, 0.018)],
        "2. Clean wall, far range (10m, range-falloff-only weak)"
    ),

    make_pulse(
        [(2.0, 0.05), (9.0, 0.50)],
        "3. Smoke-then-wall (weak near, strong far)"
    ),

    make_pulse(
        [(3.0, 0.03), (3.6, 0.018), (4.1, 0.01)],
        "4. Dust cloud (decaying, tightly spaced)"
    ),

    make_pulse(
        [],
        "5. Total absorption (dense smoke, no return at all)"
    ),

    make_pulse(
        [(5.0, 0.04)],
        "6. Low-reflectivity real surface (dark wall, 5m)"
    ),

    # --- Add your own scenarios below ---
    # make_pulse(
    #     [(distance, intensity), ...],
    #     "Description"
    # ),

]


def run_tests():
    print("=" * 78)
    print("pulse_classifier.py — Standalone Test")
    print("=" * 78)

    for pulse, label in SCENARIOS:
        features = extract_pulse_physics(pulse)
        result   = classify_pulse(pulse, features)

        print(f"\n[{label}]")
        print(f"  p_surface              : {round(result.p_surface, 3)}")
        print(f"  p_smoke                : {round(result.p_smoke, 3)}")
        print(f"  p_unknown              : {round(result.p_unknown, 3)}")
        print(f"  best_surface_echo_index: {result.best_surface_echo_index}")
        print(f"  evidence_notes         : {result.evidence_notes}")

    print("\n" + "=" * 78)
    print("Done. Review p_surface/p_smoke/p_unknown against each scenario's")
    print("intended physical meaning.")
    print("=" * 78)


if __name__ == "__main__":
    run_tests()