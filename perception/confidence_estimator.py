import numpy as np
import cv2

class ConfidenceEstimator:

    def __init__(self, window=15):
        self.window = window

    def estimate(self, frame, depth_map):

        # -------------------------
        # DEPTH CONFIDENCE MAP
        # -------------------------

        blur_depth = cv2.GaussianBlur(
            depth_map,
            (self.window, self.window),
            0
        )

        diff = (depth_map - blur_depth) ** 2

        local_var = cv2.GaussianBlur(
            diff,
            (self.window, self.window),
            0
        )

        depth_conf_map = 1.0 - (
            local_var /
            (local_var.max() + 1e-8)
        )

        depth_conf_map = np.clip(
            depth_conf_map,
            0,
            1
        )

        depth_score = float(
            np.mean(depth_conf_map)
        )

        # -------------------------
        # IMAGE QUALITY ANALYSIS
        # -------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Blur Detection
        lap_var = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        sharpness_score = min(
            lap_var / 500,
            1.0
        )

        # Brightness
        brightness = np.mean(gray)

        if 80 <= brightness <= 180:
            brightness_score = 1.0

        elif 50 <= brightness <= 220:
            brightness_score = 0.7

        else:
            brightness_score = 0.3

        # Edge Density
        edges = cv2.Canny(
            gray,
            100,
            200
        )

        edge_ratio = (
            np.count_nonzero(edges)
            / edges.size
        )

        edge_score = min(
            edge_ratio * 10,
            1.0
        )

        # -------------------------
        # FINAL CONFIDENCE SCORE
        # -------------------------

        mean_confidence = (
            0.25 * depth_score +
            0.35 * sharpness_score +
            0.20 * brightness_score +
            0.20 * edge_score
        )

        mean_confidence = float(
            np.clip(
                mean_confidence,
                0,
                1
            )
        )

        # -------------------------
        # CONFIDENCE MAP
        # -------------------------

        confidence_map = (
            depth_conf_map *
            mean_confidence
        )

        return confidence_map, mean_confidence

    def confidence_state(self, conf):

        if conf > 0.75:
            return "HIGH"

        elif conf > 0.45:
            return "MEDIUM"

        else:
            return "LOW"


if __name__ == "__main__":

    img = np.random.randint(
        0,
        255,
        (480, 640, 3),
        dtype=np.uint8
    )

    depth = np.random.rand(
        480,
        640
    ).astype(np.float32)

    ce = ConfidenceEstimator()

    conf_map, mean_conf = ce.estimate(
        img,
        depth
    )

    print("Confidence:", round(mean_conf, 3))
    print("State:", ce.confidence_state(mean_conf))
    print("Map Shape:", conf_map.shape)