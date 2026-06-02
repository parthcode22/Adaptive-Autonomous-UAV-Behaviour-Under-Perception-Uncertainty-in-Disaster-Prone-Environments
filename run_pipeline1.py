import cv2
import numpy as np

from perception.depth_estimator import DepthEstimator
from perception.confidence_estimator import ConfidenceEstimator
from perception.sensor_manager import SensorManager


# ==================================
# Initialize Modules
# ==================================

depth_estimator = DepthEstimator()
confidence_estimator = ConfidenceEstimator()
sensor_manager = SensorManager()


# ==================================
# Webcam
# ==================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


# ==================================
# Main Loop
# ==================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # ==================================
    # Depth Estimation
    # ==================================

    depth = depth_estimator.predict(frame)

    # ==================================
    # Confidence Estimation
    # ==================================

    conf_map, mean_conf = confidence_estimator.estimate(
        frame,
        depth
    )

    confidence_state = (
        confidence_estimator.confidence_state(
            mean_conf
        )
    )

    # ==================================
    # Sensor Selection
    # ==================================

    sensor = sensor_manager.choose_sensor(
        mean_conf
    )

    # ==================================
    # RGB / IR / THERMAL
    # ==================================

    if sensor == "RGB":

        display_frame = frame.copy()

    elif sensor == "IR":

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        display_frame = cv2.applyColorMap(
            gray,
            cv2.COLORMAP_BONE
        )

    else:

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        display_frame = cv2.applyColorMap(
            gray,
            cv2.COLORMAP_JET
        )

    # ==================================
    # Depth Visualization
    # ==================================

    depth_display = cv2.normalize(
        depth,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    depth_display = depth_display.astype(
        np.uint8
    )

    depth_display = cv2.applyColorMap(
        depth_display,
        cv2.COLORMAP_MAGMA
    )

    # ==================================
    # Confidence Visualization
    # ==================================

    conf_display = (
        conf_map * 255
    ).astype(np.uint8)

    conf_display = cv2.applyColorMap(
        conf_display,
        cv2.COLORMAP_VIRIDIS
    )

    # ==================================
    # Overlay Information
    # ==================================

    cv2.putText(
        display_frame,
        f"Confidence: {mean_conf:.2f}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"State: {confidence_state}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        display_frame,
        f"Sensor: {sensor}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # ==================================
    # Resize Panels
    # ==================================

    h, w = frame.shape[:2]

    depth_display = cv2.resize(
        depth_display,
        (w, h)
    )

    conf_display = cv2.resize(
        conf_display,
        (w, h)
    )

    # ==================================
    # Combine Views
    # ==================================

    combined = np.hstack(
        [
            display_frame,
            depth_display,
            conf_display
        ]
    )

    # ==================================
    # Show Output
    # ==================================

    cv2.imshow(
        "Confidence-Aware UAV Perception",
        combined
    )

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break


# ==================================
# Cleanup
# ==================================

cap.release()
cv2.destroyAllWindows()