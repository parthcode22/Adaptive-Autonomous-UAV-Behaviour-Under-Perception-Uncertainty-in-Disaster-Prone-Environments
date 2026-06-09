import cv2
import numpy as np
from ultralytics import YOLO

class DynamicObstacleDetector:
    def __init__(self, model_size="yolov8n.pt"):
        """
        model_size options:
        yolov8n.pt → nano  → fastest → best for CPU
        yolov8s.pt → small → balanced
        yolov8m.pt → medium → more accurate but slower
        We use nano because you have Intel integrated GPU
        """
        print("[YOLO] Loading model...")
        self.model = YOLO(model_size)
        print("[YOLO] Model ready.")

        # ── Classes we care about in disaster environment ──
        # COCO class IDs
        self.target_classes = {
            0: "person",      # victim / rescuer
            2: "car",         # vehicle obstacle
            3: "motorcycle",
            5: "bus",
            7: "truck",
            15: "cat",        # animal
            16: "dog",
            56: "chair",      # debris proxy
            57: "couch",
        }

        # ── Confidence threshold ──
        # Only show detections above this score
        self.detect_threshold = 0.4

        # ── Optical flow for dynamic detection ──
        self.prev_gray = None
        self.flow_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03
            )
        )

    def detect(self, frame):
        """
        Run YOLOv8 on frame.
        Returns list of detections with box, class, confidence.
        """
        results = self.model(
            frame,
            conf=self.detect_threshold,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:
            class_id = int(box.cls[0])

            if class_id not in self.target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = self.target_classes[class_id]

            detections.append({
                "label": label,
                "conf": round(conf, 3),
                "box": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "area": (x2 - x1) * (y2 - y1),
                "is_victim": False
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
        Combining both = knowing WHAT it is AND if it is moving.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0.0

        x1, y1, x2, y2 = box

        # Crop to bounding box region only
        box_curr = gray[y1:y2, x1:x2]
        box_prev = self.prev_gray[y1:y2, x1:x2]

        if box_curr.size == 0 or box_prev.size == 0:
            return False, 0.0

        # Compute dense optical flow in box region
        flow = cv2.calcOpticalFlowFarneback(
            box_prev,
            box_curr,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        # Magnitude of flow vectors
        magnitude = np.sqrt(
            flow[..., 0] ** 2 + flow[..., 1] ** 2
        )

        mean_flow = float(np.mean(magnitude))

        # If mean flow > threshold → object is moving
        is_moving = mean_flow > 2.0
        self.prev_gray = gray

        return is_moving, round(mean_flow, 3)

    def calculate_ttc(self, detection, drone_speed=1.0):
        """
        Time To Collision = Distance / Speed

        We estimate distance from bounding box area.
        Larger box = object is closer.
        WHY? As drone approaches obstacle, box gets bigger.

        distance_proxy = 1 / sqrt(area_ratio)
        where area_ratio = box_area / frame_area
        """
        frame_area = 640 * 480
        box_area = detection["area"]

        if box_area == 0:
            return float('inf')

        area_ratio = box_area / frame_area

        # Larger area = closer = smaller distance
        distance_proxy = 1.0 / (np.sqrt(area_ratio) + 1e-6)

        ttc = distance_proxy / (drone_speed + 1e-6)

        return round(ttc, 2)

    def draw(self, frame, detections, dynamic_flags):
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            is_moving = dynamic_flags[i] if i < len(dynamic_flags) else False
            ttc       = self.calculate_ttc(det)
            is_victim = det.get("is_victim", False)

            # ── Colour logic ──
            if det["label"] == "person" and is_victim:
                color = (0, 165, 255)    # orange — victim (still)
            elif det["label"] == "person" and is_moving:
                color = (0, 0, 255)      # red — rescuer/moving person
            elif is_moving:
                color = (0, 0, 255)      # red — any moving obstacle
            else:
                color = (0, 255, 0)      # green — static obstacle

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            if det["label"] == "person" and is_victim:
                status = "VICTIM"
            elif det["label"] == "person" and is_moving:
                status = "RESCUER"
            else:
                status = "MOVING" if is_moving else "STATIC"

            label_text = (
                f"{det['label']} {status} "
                f"{det['conf']:.2f} "
                f"TTC:{ttc:.1f}s"
            )
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
    cap = cv2.VideoCapture(0)

    print("Running obstacle detector... Press Q to quit")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Detect objects
        detections = detector.detect(frame)

        # Check which are dynamic
        dynamic_flags = []

        for det in detections:
            is_moving, flow = detector.is_dynamic(
                frame,
                det["box"]
            )

            dynamic_flags.append(is_moving)

            # ── Victim detection logic ──
            # Person + not moving = potential victim
            # Person + moving     = rescuer or dynamic obstacle
            if det["label"] == "person" and not is_moving:
                det["is_victim"] = True
                print(f"VICTIM DETECTED at {det['center']}")
            else:
                det["is_victim"] = False

        # Draw results
        frame = detector.draw(frame, detections, dynamic_flags)

        # Print summary
        if detections:
            print(
                f"Detected {len(detections)} objects | "
                f"Dynamic: {sum(dynamic_flags)}"
            )

        cv2.imshow("Dynamic Obstacle Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()