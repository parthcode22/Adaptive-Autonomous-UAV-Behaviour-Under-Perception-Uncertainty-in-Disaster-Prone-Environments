import cv2
import numpy as np
from ultralytics import YOLO

class DynamicObstacleDetector:
    def __init__(self, model_size="yolov8s.pt"):
        """
        model_size options:
        yolov8n.pt → nano  → fastest, lower accuracy
        yolov8s.pt → small → better accuracy, still real-time
        We use small for better detection on thermal-style frames.
        """
        print("[YOLO] Loading model...")
        self.model = YOLO(model_size)
        print("[YOLO] Model ready.")

        # ── Classes we track ──
        # 0 = person (obstacle/rescuer tracking, NOT victim logic —
        #     victim detection handled by UWB radar per locked arch)
        # Common indoor objects mapped to generic "obstacle"
        self.target_classes = {
            0  : "person",
            56 : "obstacle",   # chair
            57 : "obstacle",   # couch
            58 : "obstacle",   # potted plant
            59 : "obstacle",   # bed
            60 : "obstacle",   # dining table
            62 : "obstacle",   # tv/monitor
            63 : "obstacle",   # laptop
            64 : "obstacle",   # mouse
            67 : "obstacle",   # cell phone
            73 : "obstacle",   # book
            74 : "obstacle",   # clock
            75 : "obstacle",   # vase
        }

        # ── Confidence threshold ──
        # Lower than default — thermal-colormapped frames are
        # out-of-distribution for YOLO trained on RGB
        self.detect_threshold = 0.20

        # ── Optical flow for dynamic detection ──
        self.prev_gray   = None
        self.flow_params = dict(
            winSize   = (15, 15),
            maxLevel  = 2,
            criteria  = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10, 0.03
            )
        )

    def detect(self, frame):
        """
        Run YOLO on frame.
        Returns list of detections with box, class, confidence.
        """
        results = self.model(
            frame,
            conf    = self.detect_threshold,
            verbose = False
        )[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in self.target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            label = self.target_classes[class_id]

            detections.append({
                "label" : label,
                "conf"  : round(conf, 3),
                "box"   : (x1, y1, x2, y2),
                "center": ((x1+x2)//2, (y1+y2)//2),
                "area"  : (x2-x1) * (y2-y1)
            })

        return detections

    def is_dynamic(self, frame, box):
        """
        Check if detected object is MOVING using optical flow.
        Static object → small flow → not dynamic
        Moving object → large flow → dynamic obstacle

        WHY optical flow?
        YOLO detects objects but cannot tell if they are moving.
        Optical flow measures pixel movement between frames.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0

        x1, y1, x2, y2 = box

        box_curr = gray[y1:y2, x1:x2]
        box_prev = self.prev_gray[y1:y2, x1:x2]

        if box_curr.size == 0 or box_prev.size == 0:
            return False, 0.0

        if box_curr.shape != box_prev.shape:
            box_prev = cv2.resize(box_prev, (box_curr.shape[1], box_curr.shape[0]))

        flow = cv2.calcOpticalFlowFarneback(
            box_prev, box_curr,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        magnitude = np.sqrt(
            flow[..., 0]**2 + flow[..., 1]**2
        )
        mean_flow = float(np.mean(magnitude))

        # Lowered threshold — 0.8 instead of 2.0, more sensitive
        is_moving = mean_flow > 0.8
        self.prev_gray = gray

        return is_moving, round(mean_flow, 3)

    def calculate_ttc(self, detection, drone_speed=1.0):
        """
        Time To Collision = Distance / Speed
        Estimate distance from bounding box area.
        Larger box = object is closer.
        """
        frame_area = 640 * 480
        box_area   = detection["area"]

        if box_area == 0:
            return float('inf')

        area_ratio = box_area / frame_area
        distance_proxy = 1.0 / (np.sqrt(area_ratio) + 1e-6)

        ttc = distance_proxy / (drone_speed + 1e-6)
        return round(ttc, 2)

    def draw(self, frame, detections, dynamic_flags):
        """
        Draw bounding boxes on frame.
        Person STATIC  = green
        Person MOVING  = red
        Obstacle       = orange
        """
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            is_moving = dynamic_flags[i] if i < len(dynamic_flags) else False
            ttc       = self.calculate_ttc(det)

            if det["label"] == "person":
                color  = (0, 0, 255) if is_moving else (0, 255, 0)
                status = "MOVING" if is_moving else "STATIC"
                label_text = f"PERSON {status} {det['conf']:.2f} TTC:{ttc:.1f}s"
            else:
                color      = (0, 165, 255)
                label_text = f"OBSTACLE {det['conf']:.2f} TTC:{ttc:.1f}s"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, label_text,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
            cv2.circle(frame, det["center"], 4, color, -1)

        return frame


# ── Test with webcam ──
if __name__ == "__main__":
    detector = DynamicObstacleDetector()
    cap      = cv2.VideoCapture(0)

    print("Running obstacle detector... Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        dynamic_flags = []
        for det in detections:
            is_moving, flow = detector.is_dynamic(frame, det["box"])
            dynamic_flags.append(is_moving)

        frame = detector.draw(frame, detections, dynamic_flags)

        if detections:
            print(f"Detected {len(detections)} objects | "
                  f"Dynamic: {sum(dynamic_flags)}")

        cv2.imshow("Dynamic Obstacle Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()