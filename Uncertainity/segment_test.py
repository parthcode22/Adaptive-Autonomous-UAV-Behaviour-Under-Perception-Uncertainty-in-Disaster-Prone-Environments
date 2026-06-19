import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from Coherence_map import CoherenceMapComputer

# ─────────────────────────────────────────────
# Step 1: Segment the combined grid image into
# 5 individual disaster images
# Step 2: Run coherence map on each
# Step 3: Show comparison results
# ─────────────────────────────────────────────

class CombinedImageSegmenter:
    def __init__(self, combined_image_path):
        self.img = cv2.imread(combined_image_path)
        if self.img is None:
            raise FileNotFoundError(
                f"Could not load {combined_image_path}"
            )
        self.h, self.w = self.img.shape[:2]
        print(f"Combined image size: {self.w} x {self.h}")

    def segment(self):
        """
        Based on the grid layout:
        Row 1: [landslide] [fire] [collapse]
        Row 2: [smoke_room] [debris_field (wider)]

        We split based on relative proportions observed:
        Row 1 height ≈ 49% of total height
        Row 2 height ≈ 51% of total height

        Row 1: 3 columns roughly equal width (33% each)
        Row 2: 2 columns - smoke_room ≈ 50%, debris_field ≈ 50%
        """
        h, w = self.h, self.w

        # Row split point
        row1_h = int(h * 0.485)

        # Row 1 - 3 images
        col1_w = int(w * 0.328)
        col2_w = int(w * 0.328)

        landslide  = self.img[0:row1_h, 0:col1_w]
        fire       = self.img[0:row1_h, col1_w:col1_w+col2_w]
        collapse   = self.img[0:row1_h, col1_w+col2_w:w]

        # Row 2 - 2 images
        col1_w_r2 = int(w * 0.495)

        smoke_room   = self.img[row1_h:h, 0:col1_w_r2]
        debris_field = self.img[row1_h:h, col1_w_r2:w]

        images = {
            "landslide.jpg"        : landslide,
            "fire_building.jpg"    : fire,
            "building_collapse.jpg": collapse,
            "smoke_room.jpg"       : smoke_room,
            "debris_field.jpg"     : debris_field,
        }

        # Save each segment
        os.makedirs("test_images", exist_ok=True)
        for name, im in images.items():
            path = os.path.join("test_images", name)
            cv2.imwrite(path, im)
            print(f"Saved: {path}  shape={im.shape}")

        return images

    def visualise(self, images):
        """Show segmented images for verification"""
        n = len(images)
        fig, axes = plt.subplots(1, n, figsize=(20, 4))
        for i, (name, im) in enumerate(images.items()):
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            axes[i].imshow(rgb)
            axes[i].set_title(name, fontsize=9)
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig("segmented_check.png", dpi=100, bbox_inches='tight')
        plt.show()


class DisasterImageTester:
    def __init__(self, image_dir="test_images"):
        self.image_dir = image_dir
        self.computer   = CoherenceMapComputer()

    def process_image(self, filepath):
        img = cv2.imread(filepath)
        if img is None:
            return None, None
        img  = cv2.resize(img, (320, 240))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        result = self.computer.compute(gray)
        return img, result

    def run_all(self, files):
        n = len(files)
        fig, axes = plt.subplots(n, 5, figsize=(22, 4.5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(
            "Phase Coherence Uncertainty — Disaster Image Test",
            fontsize=16, fontweight='bold'
        )

        all_results = []

        for i, fname in enumerate(files):
            filepath = os.path.join(self.image_dir, fname)
            print(f"\nProcessing: {fname}")

            img, result = self.process_image(filepath)
            if img is None:
                continue

            all_results.append({"filename": fname, "result": result})

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i][0].imshow(img_rgb)
            axes[i][0].set_title(f"{fname}\n(Original)", fontsize=10)

            axes[i][1].imshow(
                result["coherence_map"], cmap='RdYlGn', vmin=0, vmax=1
            )
            axes[i][1].set_title("Coherence map", fontsize=10)

            im = axes[i][2].imshow(
                result["conf_map"], cmap='viridis', vmin=0, vmax=1
            )
            axes[i][2].set_title("Confidence map", fontsize=10)

            tier_colored = np.zeros(
                (*result["tier_map"].shape, 3), dtype=np.uint8
            )
            tier_colored[result["tier_map"] == 3] = [0, 255, 0]
            tier_colored[result["tier_map"] == 2] = [0, 255, 255]
            tier_colored[result["tier_map"] == 1] = [0, 165, 255]
            tier_colored[result["tier_map"] == 0] = [255, 0, 0]
            axes[i][3].imshow(tier_colored)
            axes[i][3].set_title("Tier map", fontsize=10)

            axes[i][4].axis('off')
            summary = (
                f"GLOBAL CONFIDENCE: {result['global_conf']:.4f}\n"
                f"TIER: {result['tier']}\n"
                f"Drop signal: {result['drop_signal']:.4f}\n"
                f"Danger pixels: {result['danger_ratio']*100:.1f}%\n\n"
                f"SCALE COHERENCE:\n"
                f"  Small  : {result['small_coh']:.4f}\n"
                f"  Normal : {result['normal_coh']:.4f}\n"
                f"  Large  : {result['large_coh']:.4f}\n\n"
                f"REGIONAL:\n"
                f"  Left   : {result['left_conf']:.4f}\n"
                f"  Center : {result['center_conf']:.4f}\n"
                f"  Right  : {result['right_conf']:.4f}"
            )

            tier_colors = {
                "HIGH": "lightgreen", "MODERATE": "khaki",
                "LOW": "lightsalmon", "DANGER": "lightcoral"
            }
            box_color = tier_colors.get(result['tier'], 'wheat')

            axes[i][4].text(
                0.05, 0.95, summary,
                transform=axes[i][4].transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=box_color, alpha=0.6)
            )

            for j in range(4):
                axes[i][j].axis('off')

            print(f"  Confidence: {result['global_conf']:.4f} "
                  f"| Tier: {result['tier']}")
            print(f"  Small: {result['small_coh']:.4f} "
                  f"| Normal: {result['normal_coh']:.4f} "
                  f"| Large: {result['large_coh']:.4f}")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig("disaster_test_results.png", dpi=100, bbox_inches='tight')
        plt.show()

        print("\n" + "="*80)
        print("COMPARISON TABLE")
        print("="*80)
        print(f"{'Image':<25} {'Conf':>8} {'Tier':>10} "
              f"{'Small':>8} {'Normal':>8} {'Large':>8}")
        print("-"*80)
        for r in all_results:
            res = r["result"]
            print(f"{r['filename']:<25} "
                  f"{res['global_conf']:>8.4f} "
                  f"{res['tier']:>10} "
                  f"{res['small_coh']:>8.4f} "
                  f"{res['normal_coh']:>8.4f} "
                  f"{res['large_coh']:>8.4f}")

        return all_results


# ── Run ──
if __name__ == "__main__":
    # Step 1: Segment the combined image
    # Update this path to where you saved the combined image
    combined_path = "test_images/combined.png"

    segmenter = CombinedImageSegmenter(combined_path)
    images    = segmenter.segment()
    segmenter.visualise(images)

    print("\n" + "="*60)
    print("Segmentation done. Now running coherence tests...")
    print("="*60)

    # Step 2: Run coherence tests on segmented images
    tester  = DisasterImageTester(image_dir="test_images")
    results = tester.run_all(list(images.keys()))

    # Add this for landslide specifically
    print("\nLandslide regional breakdown:")
    print(f"  Left   : {all_results[0]['result']['left_conf']:.4f}")
    print(f"  Center : {all_results[0]['result']['center_conf']:.4f}")
    print(f"  Right  : {all_results[0]['result']['right_conf']:.4f}")

    print("\nSUCCESS — Disaster image testing complete")