import cv2
import numpy as np

# ── Confidence thresholds ──
THRESHOLDS = {
    "high"   : 0.80,
    "medium" : 0.60,
    "low"    : 0.40
}

MODES = {
    "normal"  : {"speed": 1.0, "margin": 1.0},
    "reduce"  : {"speed": 0.6, "margin": 1.2},
    "cautious": {"speed": 0.3, "margin": 1.8},
    "hover"   : {"speed": 0.0, "margin": 2.5},
}

class SensorManager:

    def __init__(self):
        self.current_sensor = "rgb"

    def get_mode(self, mean_conf):
        if mean_conf > THRESHOLDS["high"]:
            return "normal"
        elif mean_conf > THRESHOLDS["medium"]:
            return "reduce"
        elif mean_conf > THRESHOLDS["low"]:
            return "cautious"
        else:
            return "hover"

    def simulate_ir(self, bgr):
        gray    = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe   = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def simulate_thermal(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

    def get_active_frame(self, bgr, mean_conf):
        mode   = self.get_mode(mean_conf)
        sensor = "rgb"

        if mean_conf <= THRESHOLDS["low"]:
            bgr    = self.simulate_thermal(bgr)
            sensor = "thermal"
        elif mean_conf <= THRESHOLDS["medium"]:
            bgr    = self.simulate_ir(bgr)
            sensor = "ir"

        return bgr, mode, sensor