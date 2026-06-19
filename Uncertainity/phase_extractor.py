import numpy as np
import cv2
import matplotlib.pyplot as plt
from Uncertainity.Gabor_filter import GaborFilterBank

class PhaseMapExtractor:
    def __init__(self):
        """
        Takes the 12 Gabor responses and extracts:

        1. Phase map per scale (3 maps)
           → average phase across 4 orientations at each scale
           → tells us dominant structural direction at each scale

        2. Magnitude map per scale (3 maps)
           → average magnitude across 4 orientations at each scale
           → tells us HOW MUCH structure exists at each scale

        3. Scale energy ratio
           → large_magnitude / (small_magnitude + 1e-8)
           → in clear scene: ratio is HIGH (38x as we saw)
           → in smoke/dark : ratio COLLAPSES
           → this collapse = uncertainty signal
        """
        self.bank = GaborFilterBank()

    def extract(self, gray_image):
        """
        Input : grayscale float32 image normalised 0-1
        Output: dict containing all phase and magnitude maps
        """
        # Get all 12 filter responses
        responses = self.bank.apply(gray_image)

        # ── Group by scale ──
        scales = {
            "small" : [r for r in responses if r["scale"] == "small"],
            "normal": [r for r in responses if r["scale"] == "normal"],
            "large" : [r for r in responses if r["scale"] == "large"],
        }

        result = {}

        for scale_name, scale_responses in scales.items():

            # Stack all 4 orientation magnitudes
            mags = np.stack(
                [r["magnitude"] for r in scale_responses],
                axis=0
            )  # shape: [4, H, W]

            # Stack all 4 orientation phases
            phases = np.stack(
                [r["phase"] for r in scale_responses],
                axis=0
            )  # shape: [4, H, W]

            # ── Mean magnitude map ──
            # Average response strength across all orientations
            mean_mag = np.mean(mags, axis=0)

            # ── Dominant phase map ──
            # At each pixel, which orientation responded STRONGEST?
            # That orientation's phase = dominant phase
            dominant_idx   = np.argmax(mags, axis=0)
            dominant_phase = np.zeros_like(phases[0])

            for i in range(4):
                mask = (dominant_idx == i)
                dominant_phase[mask] = phases[i][mask]

            # ── Phase consistency map ──
            # How much do all 4 orientations AGREE on phase?
            # Use circular mean — phases are angles (-π to π)
            # exp(i×phase) converts to unit vector on unit circle
            # Mean of unit vectors → length = coherence
            # Length near 1 = all orientations agree
            # Length near 0 = all orientations disagree

            complex_phases = np.exp(1j * phases)
            # shape: [4, H, W] complex

            mean_complex   = np.mean(complex_phases, axis=0)
            # shape: [H, W] complex

            phase_consistency = np.abs(mean_complex)
            # shape: [H, W] float, range 0-1
            # 1 = perfect agreement = coherent = confident
            # 0 = random phases    = incoherent = uncertain

            result[scale_name] = {
                "mean_magnitude"    : mean_mag,
                "dominant_phase"    : dominant_phase,
                "phase_consistency" : phase_consistency,
            }

        # ── Scale energy ratio ──
        small_energy  = np.mean(result["small"]["mean_magnitude"])
        normal_energy = np.mean(result["normal"]["mean_magnitude"])
        large_energy  = np.mean(result["large"]["mean_magnitude"])

        total_energy  = small_energy + normal_energy + large_energy + 1e-8

        result["scale_energy"] = {
            "small"        : small_energy,
            "normal"       : normal_energy,
            "large"        : large_energy,
            "total"        : total_energy,
            "small_ratio"  : small_energy  / total_energy,
            "normal_ratio" : normal_energy / total_energy,
            "large_ratio"  : large_energy  / total_energy,
        }
        
        result["fine_detail_score"] = float(
            np.clip(
                result["scale_energy"]["small_ratio"] * 3.0,
                0.0,
                1.0
            )
        )

        # ── Global phase coherence score ──
        # Weighted average of phase consistency across all scales
        # Small scale consistency weighted MORE
        # WHY? Fine details are first to disappear in smoke
        # If small scale is still coherent → scene is very clear

        small_coh  = np.mean(result["small"]["phase_consistency"])
        normal_coh = np.mean(result["normal"]["phase_consistency"])
        large_coh  = np.mean(result["large"]["phase_consistency"])

        # Weights: small=0.5, normal=0.3, large=0.2
        # Small scale coherence matters most for confidence
        global_coherence = (
            0.5 * small_coh  +
            0.3 * normal_coh +
            0.2 * large_coh
        )

        result["global_coherence"] = round(float(global_coherence), 4)

        # ── Confidence from coherence ──
        # Scale coherence 0-1 to our confidence tier system
        # Raw coherence tends to be in 0.3-0.8 range
        # Stretch and clip to use full 0-1 range
        confidence = float(np.clip(
            (global_coherence - 0.3) / 0.5,
            0.0, 1.0
        ))
        result["confidence"] = round(confidence, 4)
        
        print("Fine detail received:", result["fine_detail_score"])

        return result

    def visualise(self, gray_image, result):
        """
        Shows phase maps and coherence maps for all 3 scales.
        4 columns: magnitude | dominant phase | coherence | overlay
        """
        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        fig.suptitle(
            f"Phase map extraction results\n"
            f"Global coherence: {result['global_coherence']:.4f} | "
            f"Confidence: {result['confidence']:.4f}",
            fontsize=13
        )

        scale_names = ["small (σ=2)", "normal (σ=4)", "large (σ=8)"]
        col_titles  = [
            "Mean magnitude",
            "Dominant phase",
            "Phase coherence",
            "Confidence overlay"
        ]

        for col, title in enumerate(col_titles):
            axes[0][col].set_title(title, fontsize=10, fontweight='bold')

        for i, scale in enumerate(["small", "normal", "large"]):
            data = result[scale]

            # Column 0 — magnitude
            axes[i][0].imshow(
                data["mean_magnitude"], cmap="viridis"
            )
            axes[i][0].set_ylabel(
                scale_names[i], fontsize=9, fontweight='bold'
            )

            # Column 1 — dominant phase (-π to π)
            axes[i][1].imshow(
                data["dominant_phase"],
                cmap="hsv",        # HSV colormap = circular colours
                vmin=-np.pi,
                vmax=np.pi
            )

            # Column 2 — phase coherence (0=uncertain, 1=confident)
            im = axes[i][2].imshow(
                data["phase_consistency"],
                cmap="RdYlGn",     # Red=uncertain, Green=confident
                vmin=0, vmax=1
            )
            plt.colorbar(im, ax=axes[i][2], fraction=0.046)

            # Column 3 — overlay: original image + coherence
            overlay = cv2.applyColorMap(
                (data["phase_consistency"] * 255).astype(np.uint8),
                cv2.COLORMAP_JET
            )
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            orig_rgb    = cv2.cvtColor(
                (gray_image * 255).astype(np.uint8),
                cv2.COLOR_GRAY2RGB
            )
            blended = cv2.addWeighted(orig_rgb, 0.6, overlay_rgb, 0.4, 0)
            axes[i][3].imshow(blended)

            for j in range(4):
                axes[i][j].axis("off")

        # Energy ratio bar
        energies = result["scale_energy"]
        fig.text(
            0.5, 0.02,
            f"Scale energy ratio — "
            f"small: {energies['small_ratio']:.3f} | "
            f"normal: {energies['normal_ratio']:.3f} | "
            f"large: {energies['large_ratio']:.3f}",
            ha="center", fontsize=10
        )

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        plt.show()


# ── Test ──
if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 240))
        gray = gray.astype(np.float32) / 255.0

        extractor = PhaseMapExtractor()
        result    = extractor.extract(gray)

        print("\n── Phase extraction results ──")
        print(f"Global coherence : {result['global_coherence']}")
        print(f"Confidence score : {result['confidence']}")
        print()
        print("Scale energy breakdown:")
        e = result["scale_energy"]
        print(f"  Small  energy : {e['small']:.4f}  "
              f"({e['small_ratio']*100:.1f}%)")
        print(f"  Normal energy : {e['normal']:.4f}  "
              f"({e['normal_ratio']*100:.1f}%)")
        print(f"  Large  energy : {e['large']:.4f}  "
              f"({e['large_ratio']*100:.1f}%)")
        print()
        print("Per scale coherence:")
        for scale in ["small", "normal", "large"]:
            coh = np.mean(result[scale]["phase_consistency"])
            print(f"  {scale:6s} coherence : {coh:.4f}")

        extractor.visualise(gray, result)
        print("\nSUCCESS — Phase map extraction working")