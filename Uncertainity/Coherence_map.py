import numpy as np
import cv2
import matplotlib.pyplot as plt

try:
    from Uncertainity.phase_extractor import PhaseMapExtractor
except ImportError:
    from phase_extractor import PhaseMapExtractor


class CoherenceMapComputer:
    def __init__(self):
        self.extractor = PhaseMapExtractor()

        self.weights = {
            "small" : 0.50,
            "normal": 0.35,
            "large" : 0.15
        }

        # Reference energy for "informative" scene (calibrated
        # from clear-room baseline). Used to score whether
        # ANY structure exists — fixes smoke=high-conf bug.
        self.ENERGY_REF = 0.4

        # Single threshold — SAFE vs DANGER only
        self.SAFE_THRESHOLD = 0.45

    def compute(self, gray_image):
        phase_result = self.extractor.extract(gray_image)

        small_coh  = phase_result["small"]["phase_consistency"]
        normal_coh = phase_result["normal"]["phase_consistency"]
        large_coh  = phase_result["large"]["phase_consistency"]

        H, W = gray_image.shape
        small_coh  = cv2.resize(small_coh,  (W, H))
        normal_coh = cv2.resize(normal_coh, (W, H))
        large_coh  = cv2.resize(large_coh,  (W, H))

        raw_coherence = (
            self.weights["small"]  * small_coh  +
            self.weights["normal"] * normal_coh +
            self.weights["large"]  * large_coh
        )
        global_coherence = float(np.clip(np.mean(raw_coherence), 0, 1))

        # ── Energy-based informativeness (primary signal) ──
        e = phase_result["scale_energy"]
        energy = (
            self.weights["small"]  * e["small"]  +
            self.weights["normal"] * e["normal"] +
            self.weights["large"]  * e["large"]
        )
        energy_score = float(np.clip(energy / self.ENERGY_REF, 0, 1))

        # ── Edge density ──
        gray_u8 = (gray_image * 255).astype(np.uint8)
        edges   = cv2.Canny(gray_u8, 50, 150)
        edge_density = np.count_nonzero(edges) / edges.size
        edge_score   = float(np.clip(edge_density / 0.10, 0, 1))

        # ── Contrast ──
        contrast = float(np.std(gray_image))
        contrast_score = float(np.clip(contrast / 0.20, 0, 1))

        # ── Informativeness (is there ANYTHING to navigate by) ──
        informativeness = (
            0.5 * energy_score +
            0.25 * edge_score +
            0.25 * contrast_score
        )

        # ── Final confidence: informativeness gated, coherence
        # only adds bonus when structure IS organized ──
        final_conf = float(np.clip(
            informativeness * (0.7 + 0.3 * global_coherence),
            0.0, 1.0
        ))

        # ── Pixel map: scale local coherence by global info level ──
        smooth_coherence = cv2.GaussianBlur(
            raw_coherence.astype(np.float32), (15, 15), 0
        )
        conf_map = np.clip(smooth_coherence * informativeness, 0.0, 1.0)

        # ── Binary tier map: SAFE=1, DANGER=0 ──
        tier_map = (conf_map >= self.SAFE_THRESHOLD).astype(np.uint8)

        # ── Regional scores — SAME formula as global ──
        small_mag  = cv2.resize(phase_result["small"]["mean_magnitude"],  (W, H))
        normal_mag = cv2.resize(phase_result["normal"]["mean_magnitude"], (W, H))
        large_mag  = cv2.resize(phase_result["large"]["mean_magnitude"],  (W, H))
        combined_energy_map = (
            self.weights["small"]  * small_mag +
            self.weights["normal"] * normal_mag +
            self.weights["large"]  * large_mag
        )
        edges_full = edges.astype(np.float32) / 255.0

        def regional_score(x0, x1):
            e_r   = float(np.mean(combined_energy_map[:, x0:x1]))
            es_r  = float(np.clip(e_r / self.ENERGY_REF, 0, 1))
            ed_r  = float(np.mean(edges_full[:, x0:x1]))
            eds_r = float(np.clip(ed_r / 0.10, 0, 1))
            cs_r  = float(np.clip(np.std(gray_image[:, x0:x1]) / 0.20, 0, 1))
            info_r = 0.5*es_r + 0.25*eds_r + 0.25*cs_r
            coh_r  = float(np.mean(smooth_coherence[:, x0:x1]))
            return float(np.clip(info_r * (0.7 + 0.3*coh_r), 0, 1))

        w = conf_map.shape[1]
        left_conf   = regional_score(0, w//3)
        center_conf = regional_score(w//3, 2*w//3)
        right_conf  = regional_score(2*w//3, w)

        danger_pixels = int(np.sum(tier_map == 0))
        danger_ratio  = danger_pixels / tier_map.size

        tier = self.get_tier_name(final_conf)

        return {
            "coherence_map"  : smooth_coherence,
            "conf_map"       : conf_map,
            "tier_map"       : tier_map,
            "global_conf"    : round(final_conf, 4),
            "raw_coherence"  : round(global_coherence, 4),
            "informativeness": round(informativeness, 4),
            "energy_score"   : round(energy_score, 4),
            "edge_score"     : round(edge_score, 4),
            "contrast_score" : round(contrast_score, 4),
            "left_conf"      : round(left_conf, 4),
            "center_conf"    : round(center_conf, 4),
            "right_conf"     : round(right_conf, 4),
            "danger_ratio"   : round(danger_ratio, 4),
            "tier"           : tier,
            "small_coh"      : round(float(np.mean(small_coh)), 4),
            "normal_coh"     : round(float(np.mean(normal_coh)), 4),
            "large_coh"      : round(float(np.mean(large_coh)), 4),
        }

    def get_tier_name(self, conf):
        return "SAFE" if conf >= self.SAFE_THRESHOLD else "DANGER"

    def get_colour(self, tier):
        return (0, 255, 0) if tier == "SAFE" else (0, 0, 255)

    def visualise(self, gray_image, result):
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(
            f"Coherence map — confidence: {result['global_conf']} "
            f"| tier: {result['tier']} "
            f"| danger: {result['danger_ratio']*100:.1f}%",
            fontsize=13, fontweight='bold'
        )

        # Original
        axes[0][0].imshow(gray_image, cmap='gray')
        axes[0][0].set_title("Original frame")

        # Raw coherence map
        axes[0][1].imshow(
            result["coherence_map"],
            cmap='RdYlGn', vmin=0, vmax=1
        )
        axes[0][1].set_title("Coherence map\nred=uncertain green=confident")

        # Confidence map
        im = axes[0][2].imshow(
            result["conf_map"],
            cmap='viridis', vmin=0, vmax=1
        )
        axes[0][2].set_title("Confidence map C(x,y)")
        plt.colorbar(im, ax=axes[0][2], fraction=0.046)

        # Tier map — binary SAFE/DANGER
        tier_colored = np.zeros((*result["tier_map"].shape, 3), dtype=np.uint8)
        tier_colored[result["tier_map"] == 1] = [0, 255, 0]   # SAFE
        tier_colored[result["tier_map"] == 0] = [255, 0, 0]   # DANGER
        axes[1][0].imshow(tier_colored)
        axes[1][0].set_title("Tier map\nGreen=SAFE Red=DANGER")

        # Regional confidence bar
        regions = [
            result["left_conf"],
            result["center_conf"],
            result["right_conf"]
        ]
        colours = []
        for r in regions:
            t = self.get_tier_name(r)
            c = self.get_colour(t)
            colours.append(tuple(v/255 for v in reversed(c)))

        axes[1][1].bar(
            ["Left", "Center", "Right"],
            regions,
            color=colours,
            edgecolor='gray'
        )
        axes[1][1].set_ylim(0, 1)
        axes[1][1].axhline(
            y=self.SAFE_THRESHOLD, color='green',
            linestyle='--', alpha=0.5, label='SAFE threshold'
        )
        axes[1][1].legend(fontsize=8)
        axes[1][1].set_title("Regional confidence\n(UAV navigation zones)")

        # Summary text
        axes[1][2].axis('off')
        summary = (
            f"COHERENCE ANALYSIS SUMMARY\n\n"
            f"Global confidence : {result['global_conf']}\n"
            f"Tier              : {result['tier']}\n"
            f"Informativeness   : {result['informativeness']}\n\n"
            f"Scale coherence:\n"
            f"  Small  : {result['small_coh']}\n"
            f"  Normal : {result['normal_coh']}\n"
            f"  Large  : {result['large_coh']}\n\n"
            f"Regional:\n"
            f"  Left   : {result['left_conf']}\n"
            f"  Center : {result['center_conf']}\n"
            f"  Right  : {result['right_conf']}\n\n"
            f"Danger pixels : {result['danger_ratio']*100:.1f}%"
        )
        axes[1][2].text(
            0.1, 0.9, summary,
            transform=axes[1][2].transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')
        axes[1][1].axis('on')

        plt.tight_layout()
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

        computer = CoherenceMapComputer()
        result   = computer.compute(gray)

        print("\n── Coherence map results ──")
        print(f"Global confidence : {result['global_conf']}")
        print(f"Tier              : {result['tier']}")
        print(f"Informativeness   : {result['informativeness']}")
        print(f"Danger pixels     : {result['danger_ratio']*100:.1f}%")
        print()
        print("Regional confidence:")
        print(f"  Left   : {result['left_conf']}")
        print(f"  Center : {result['center_conf']}")
        print(f"  Right  : {result['right_conf']}")
        print()
        print("Scale coherence:")
        print(f"  Small  : {result['small_coh']}")
        print(f"  Normal : {result['normal_coh']}")
        print(f"  Large  : {result['large_coh']}")

        computer.visualise(gray, result)
        print("\nSUCCESS — Coherence map working")