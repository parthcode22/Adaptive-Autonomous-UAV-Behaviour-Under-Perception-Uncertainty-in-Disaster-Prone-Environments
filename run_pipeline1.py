import cv2
import numpy as np
import sys
import os

# ── Add project root to path ──
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perception.depth_estimator      import DepthEstimator
from perception.confidence_estimator import ConfidenceEstimator
from perception.sensor_manager       import SensorManager
from Objectdetection.detector        import DynamicObstacleDetector

class UAVPipeline:
    def __init__(self):
        print("[Pipeline] Loading all modules...")
        self.depth_est   = DepthEstimator()
        self.conf_est    = ConfidenceEstimator()
        self.sensor_mgr  = SensorManager()
        self.detector    = DynamicObstacleDetector()

        self.conf_history   = []
        self.switch_counter = 0
        self.SWITCH_WINDOW  = 45

        print("[Pipeline] All modules ready.")

    def laplacian_confidence(self, frame):
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap      = cv2.Laplacian(gray, cv2.CV_64F)
        variance = lap.var()
        conf     = float(np.clip(variance / 3000.0, 0.0, 1.0))
        return conf

    def get_tier(self, conf):
        if conf   >= 0.80: return "HIGH",     (0, 255, 0)
        elif conf >= 0.50: return "MODERATE", (0, 255, 255)
        elif conf >= 0.30: return "LOW",      (0, 165, 255)
        else:              return "DANGER",   (0, 0, 255)

    def run(self):
        cap           = cv2.VideoCapture(0)
        active_sensor = "rgb"
        switch_counter = 0

        print("[Pipeline] Running... Press Q to quit")
        print("-" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ── Step 1: Resize ──
            frame = cv2.resize(frame, (640, 480))

            # ── Step 2: Laplacian confidence ──
            lap_conf = self.laplacian_confidence(frame)
            self.conf_history.append(lap_conf)
            if len(self.conf_history) > self.SWITCH_WINDOW:
                self.conf_history.pop(0)
            mean_conf = float(np.mean(self.conf_history))

            # ── Step 3: Sensor switching ──
            if active_sensor == "rgb":
                if mean_conf < 0.30:
                    switch_counter += 1
                    if switch_counter >= self.SWITCH_WINDOW:
                        active_sensor  = "ir"
                        switch_counter = 0
                        print("[SWITCH] RGB → IR")
                else:
                    switch_counter = 0

            elif active_sensor == "ir":
                if mean_conf < 0.30:
                    switch_counter += 1
                    if switch_counter >= self.SWITCH_WINDOW:
                        active_sensor  = "thermal"
                        switch_counter = 0
                        print("[SWITCH] IR → Thermal")
                elif mean_conf > 0.50:
                    active_sensor  = "rgb"
                    switch_counter = 0
                    print("[RECOVER] IR → RGB")

            # ── Step 4: Apply sensor simulation ──
            active_frame, mode, sensor = \
                self.sensor_mgr.get_active_frame(frame, mean_conf)

            # ── Step 5: MiDaS depth ──
            depth = self.depth_est.predict(active_frame)

            # ── Step 6: Confidence from frame + depth ──
            conf_map, mean_conf_depth = self.conf_est.estimate(
                active_frame, depth
            )

            # Regional confidence
            w = conf_map.shape[1]
            regions = {
                "left"  : float(np.mean(conf_map[:, :w//3])),
                "center": float(np.mean(conf_map[:, w//3:2*w//3])),
                "right" : float(np.mean(conf_map[:, 2*w//3:]))
            }

            # ── Step 7: YOLO detection ──
            detections    = self.detector.detect(frame)
            dynamic_flags = []
            victims       = []

            for det in detections:
                is_moving, flow = self.detector.is_dynamic(
                    frame, det["box"]
                )
                dynamic_flags.append(is_moving)

                if det["label"] == "person" and not is_moving:
                    det["is_victim"] = True
                    victims.append(det["center"])
                else:
                    det["is_victim"] = False

            frame_det = self.detector.draw(
                frame.copy(), detections, dynamic_flags
            )

            # ── Step 8: Depth + confidence visualisation ──
            d_vis = cv2.normalize(
                depth, None, 0, 255, cv2.NORM_MINMAX
            ).astype('uint8')
            d_col = cv2.applyColorMap(d_vis, cv2.COLORMAP_MAGMA)
            c_vis = (conf_map * 255).astype('uint8')
            c_col = cv2.applyColorMap(c_vis, cv2.COLORMAP_VIRIDIS)

            d_col = cv2.resize(d_col, (640, 480))
            c_col = cv2.resize(c_col, (640, 480))

            # ── Step 9: HUD overlay ──
            tier, color = self.get_tier(mean_conf)

            cv2.putText(frame_det,
                f"Sensor: {active_sensor.upper()} | "
                f"Conf: {mean_conf:.3f} | "
                f"Tier: {tier}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)

            cv2.putText(frame_det,
                f"Objects: {len(detections)} | "
                f"Dynamic: {sum(dynamic_flags)} | "
                f"Victims: {len(victims)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2)

            cv2.putText(frame_det,
                f"L:{regions['left']:.2f} | "
                f"C:{regions['center']:.2f} | "
                f"R:{regions['right']:.2f}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

            cv2.putText(d_col,
                "DEPTH MAP",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

            cv2.putText(c_col,
                f"CONFIDENCE | Mean: {mean_conf_depth:.3f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2)

            # ── Step 10: Stack all views ──
            top     = np.hstack([frame_det, d_col])
            bottom  = np.hstack([c_col,
                self.info_panel(
                    mean_conf, tier, color,
                    active_sensor, detections,
                    dynamic_flags, victims
                )
            ])
            display = np.vstack([top, bottom])
            display = cv2.resize(display, (1280, 720))

            cv2.imshow("UAV Perception Pipeline", display)

            if victims:
                print(f"VICTIM at {victims}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def info_panel(self, conf, tier, color,
                   sensor, detections, dynamic_flags, victims):
        panel      = np.zeros((480, 640, 3), dtype=np.uint8)
        panel[:]   = (30, 30, 30)

        lines = [
            ("UAV PERCEPTION SYSTEM",          (255, 255, 255)),
            ("",                                (255, 255, 255)),
            (f"Active sensor : {sensor.upper()}", color),
            (f"Confidence    : {conf:.3f}",       color),
            (f"Tier          : {tier}",            color),
            ("",                                (255, 255, 255)),
            (f"Objects found : {len(detections)}", (200, 200, 200)),
            (f"Dynamic obs   : {sum(dynamic_flags)}", (0, 0, 255)),
            (f"Victims found : {len(victims)}",   (0, 165, 255)),
            ("",                                (255, 255, 255)),
            ("Confidence tiers:",               (200, 200, 200)),
            ("HIGH     0.80-1.00 Normal flight",(0, 255, 0)),
            ("MODERATE 0.50-0.79 Reduce speed", (0, 255, 255)),
            ("LOW      0.30-0.49 Safety margin",(0, 165, 255)),
            ("DANGER   0.00-0.29 Switch sensor",(0, 0, 255)),
            ("",                                (255, 255, 255)),
            ("Orange = Victim",                 (0, 165, 255)),
            ("Red    = Dynamic obstacle",       (0, 0, 255)),
            ("Green  = Static obstacle",        (0, 255, 0)),
        ]

        y = 30
        for text, col in lines:
            cv2.putText(
                panel, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, col, 1
            )
            y += 28

        return panel


# ── Run ──
if __name__ == "__main__":
    pipeline = UAVPipeline()
    pipeline.run()