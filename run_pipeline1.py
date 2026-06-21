import cv2
import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from perception.depth_estimator      import DepthEstimator
from perception.sensor_manager       import SensorManager
from Victimdetection.detector        import DynamicObstacleDetector
from Uncertainity.Coherence_map      import CoherenceMapComputer
from fusion_1.fusion_pipeline        import FusionPipeline, FusionPipelineConfig
from fusion_1.fusion_engine          import GaborPipelineOutput, OdometryInput, LidarPipelineOutput
from lidar.lidar_pipeline            import LidarPipeline, LidarPipelineConfig, to_fusion_output
from lidar.echo_physics              import EchoReturn, LidarPulse


class UAVPipeline:
    def __init__(self):
        print("[Pipeline] Loading all modules...")
        self.depth_est   = DepthEstimator()
        self.sensor_mgr  = SensorManager()
        self.detector    = DynamicObstacleDetector()
        self.coherence   = CoherenceMapComputer()

        # ── Fusion layer ──
        # lidar_available=True → real LidarPipelineOutput now feeds in,
        # built from placeholder pulses until a real sensor/Gazebo feed
        # is connected (see _generate_placeholder_pulses below).
        fusion_cfg       = FusionPipelineConfig(
            lidar_available  = True,
            log_to_console   = False,
            history_length   = 30,
        )
        self.fusion      = FusionPipeline(config=fusion_cfg)

        # ── LiDAR module ──
        # Stateful pipeline — holds TemporalTracker history across scans.
        self.lidar_pipeline = LidarPipeline(
            config=LidarPipelineConfig(log_to_console=False)
        )

        print("[Pipeline] All modules ready.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_gabor_output(self, coh_result) -> GaborPipelineOutput:
        """
        Convert CoherenceMapComputer output dict into a GaborPipelineOutput
        object that the fusion layer understands.
        """
        global_conf = float(coh_result.get("global_conf",  0.0))
        info        = float(coh_result.get("informativeness", 0.0))
        tier        = "SAFE" if global_conf >= 0.45 else "DANGER"

        return GaborPipelineOutput(
            global_confidence      = global_conf,
            informativeness        = info,
            tier                   = tier,
            scale_coherence_small  = float(coh_result.get("small_coh",  0.5)),
            scale_coherence_normal = float(coh_result.get("normal_coh", 0.5)),
            scale_coherence_large  = float(coh_result.get("large_coh",  0.5)),
            regional_left          = float(coh_result.get("left_conf",  0.5)),
            regional_center        = float(coh_result.get("center_conf",0.5)),
            regional_right         = float(coh_result.get("right_conf", 0.5)),
            danger_pixel_fraction  = float(coh_result.get("danger_ratio",0.0)),
        )

    def _generate_placeholder_pulses(self, gabor_confidence: float) -> list:
        """
        TEMPORARY: generates placeholder LidarPulse objects until a real
        LiDAR sensor / Gazebo feed is connected. Pulse behavior is loosely
        tied to gabor_confidence so the placeholder isn't fully disconnected
        from the rest of the scene — clearer visual scenes simulate a
        clean wall-like return, smokier scenes simulate degraded/absorbed
        returns. Replace this method entirely once real sensor data exists.
        """
        rng = np.random.default_rng()

        pulses = []
        for az in np.linspace(-0.04, 0.04, 10):
            for el in np.linspace(-0.02, 0.02, 4):
                if gabor_confidence > 0.5:
                    # Clearer scene -> simulate a clean wall-like return
                    d = 5.0 + rng.normal(0, 0.05)
                    echoes = [EchoReturn(distance=d, intensity=0.4, echo_index=0, total_echoes=1)]
                else:
                    # Smokier scene -> simulate degraded/absorbed returns
                    if rng.random() < 0.5:
                        echoes = []  # total absorption
                    else:
                        d = rng.uniform(1.0, 3.0)
                        i = rng.uniform(0.05, 0.15)
                        echoes = [EchoReturn(distance=d, intensity=i, echo_index=0, total_echoes=1)]
                pulses.append(LidarPulse(
                    echoes=echoes, beam_azimuth=az, beam_elevation=el, timestamp=time.time()
                ))
        return pulses

    def get_tier(self, conf):
        if conf >= 0.45:
            return "SAFE", (0, 255, 0)
        else:
            return "DANGER", (0, 0, 255)

    def _nav_state_color(self, nav_state: str):
        """Map NavigationState string to BGR color for display."""
        return {
            "SAFE"    : (0, 255, 0),
            "CAUTION" : (0, 200, 255),
            "DANGER"  : (0, 0, 255),
            "BLIND"   : (0, 0, 180),
        }.get(nav_state, (200, 200, 200))

    # ------------------------------------------------------------------
    # Metrics panel — now shows fusion UPS fields + LiDAR fields
    # ------------------------------------------------------------------

    def metrics_panel(self, result, tier, color,
                      detections, dynamic_flags, fps,
                      fusion_metrics: dict):

        panel    = np.zeros((360, 640, 3), dtype=np.uint8)
        panel[:] = (20, 20, 20)

        # ── Title ──
        cv2.rectangle(panel, (0, 0), (640, 34), (40, 40, 40), -1)
        cv2.putText(panel, "UAV PERCEPTION METRICS",
            (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1)

        y = 48

        # ── Section 1: Fusion State ──────────────────────────────────
        cv2.putText(panel, "FUSION STATE",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (150, 150, 150), 1)
        cv2.line(panel, (10, y+6), (630, y+6), (50, 50, 50), 1)
        y += 18

        nav_state  = fusion_metrics.get("nav_state",       "BLIND")
        dom_sensor = fusion_metrics.get("dominant_sensor", "NONE")
        nav_color  = self._nav_state_color(nav_state)

        fusion_lines = [
            (f"Nav State      : {nav_state}",
             nav_color),
            (f"Dominant Sensor: {dom_sensor}",
             (200, 200, 200)),
            (f"Scene Conf     : {fusion_metrics.get('scene_confidence',    0.0):.3f}   "
             f"Obstacle: {fusion_metrics.get('obstacle_confidence', 0.0):.3f}",
             (200, 200, 200)),
            (f"Smoke Density  : {fusion_metrics.get('smoke_density',       0.0):.3f}   "
             f"DTimer: {fusion_metrics.get('danger_timer', 0.0):.1f}s",
             (0, 180, 255) if fusion_metrics.get("smoke_density", 0) > 0.5
             else (200, 200, 200)),
            (f"W_visual       : {fusion_metrics.get('w_visual', 0.5):.3f}   "
             f"W_lidar: {fusion_metrics.get('w_lidar', 0.5):.3f}",
             (180, 180, 100)),
        ]
        for text, col in fusion_lines:
            cv2.putText(panel, text, (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
            y += 16

        y += 4

        # ── Section 2: Confidence (Gabor) ────────────────────────────
        cv2.putText(panel, "GABOR CONFIDENCE",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (150, 150, 150), 1)
        cv2.line(panel, (10, y+6), (630, y+6), (50, 50, 50), 1)
        y += 18

        conf_lines = [
            (f"Global Visibility: {result['global_conf']:.4f}", color),
            (f"Tier             : {tier}", color),
            (f"Informativeness  : {result['informativeness']:.4f}", (200, 200, 200)),
            (f"Danger Pixels    : {result['danger_ratio']*100:.1f}%",
             (0, 0, 255) if result['danger_ratio'] > 0.5 else (200, 200, 200)),
        ]
        for text, col in conf_lines:
            cv2.putText(panel, text, (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
            y += 16

        y += 4

        # ── Section 3: Scale Coherence ───────────────────────────────
        cv2.putText(panel, "SCALE COHERENCE",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (150, 150, 150), 1)
        cv2.line(panel, (10, y+6), (630, y+6), (50, 50, 50), 1)
        y += 18

        scales = [
            (f"Small  (fine)  : {result['small_coh']:.4f}",  (0, 200, 200)),
            (f"Normal (medium): {result['normal_coh']:.4f}", (0, 200, 200)),
            (f"Large  (coarse): {result['large_coh']:.4f}",  (100, 100, 200)),
        ]
        for text, col in scales:
            cv2.putText(panel, text, (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
            y += 16

        y += 4

        # ── Section 4: Regional Navigation ──────────────────────────
        cv2.putText(panel, "REGIONAL NAVIGATION",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (150, 150, 150), 1)
        cv2.line(panel, (10, y+6), (630, y+6), (50, 50, 50), 1)
        y += 18

        regions = {
            "Left"  : result["left_conf"],
            "Center": result["center_conf"],
            "Right" : result["right_conf"],
        }
        for name, val in regions.items():
            t = "SAFE" if val >= 0.45 else "DANGER"
            c = (0, 255, 0) if t == "SAFE" else (0, 0, 255)
            cv2.putText(panel,
                f"{name:6s}: {val:.3f}  [{t}]",
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)
            y += 16

        y += 4

        # ── Section 5: LiDAR (NEW) ───────────────────────────────────
        cv2.putText(panel, "LIDAR (multi-echo)",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (150, 150, 150), 1)
        cv2.line(panel, (10, y+6), (630, y+6), (50, 50, 50), 1)
        y += 18

        lidar_lines = [
            (f"p_surface: {fusion_metrics.get('lidar_p_surface', 0.0):.3f}   "
             f"p_smoke: {fusion_metrics.get('lidar_p_smoke', 0.0):.3f}",
             (200, 200, 200)),
            (f"spatial_consistency: {fusion_metrics.get('lidar_spatial', 0.0):.3f}   "
             f"valid_echo: {fusion_metrics.get('lidar_valid_echo', 0.0):.3f}",
             (200, 200, 200)),
            (f"beta_estimate: {fusion_metrics.get('lidar_beta', 0.0):.3f}   "
             f"obstacle_prox: {fusion_metrics.get('lidar_obstacle_prox', 0.0):.3f}",
             (180, 180, 100)),
        ]
        for text, col in lidar_lines:
            cv2.putText(panel, text, (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
            y += 16

        y += 4

        # ── Section 6: Detection ─────────────────────────────────────
        cv2.putText(panel, "DETECTION",
            (10, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.42, (150, 150, 150), 1)
        cv2.line(panel, (10, y+6), (630, y+6), (50, 50, 50), 1)
        y += 18

        persons   = [d for d in detections if d["label"] == "person"]
        obstacles = [d for d in detections if d["label"] == "obstacle"]
        moving    = sum(dynamic_flags)

        cv2.putText(panel,
            f"Persons: {len(persons)}  Moving: {moving}  Obstacles: {len(obstacles)}",
            (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        y += 16

        cv2.putText(panel,
            f"FPS: {fps:.1f}   Radar: {fusion_metrics.get('radar_flag', 0)}   "
            f"Fusion FPS: {fusion_metrics.get('fusion_fps', 0.0):.1f}",
            (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)
        y += 16

        # ── Legend ───────────────────────────────────────────────────
        cv2.line(panel, (10, 342), (630, 342), (50, 50, 50), 1)
        cv2.putText(panel, "SAFE conf>=0.45",
            (10, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
        cv2.putText(panel, "DANGER conf<0.45",
            (220, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255), 1)
        cv2.putText(panel, f"Cycle #{fusion_metrics.get('fusion_cycle', 0)}",
            (450, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1)

        return panel

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        cap = cv2.VideoCapture(0)

        PROCESS_EVERY = 8
        frame_count   = 0
        cached        = None
        prev_t        = time.time()

        # initialise fusion_metrics so metrics_panel never crashes on
        # the first few frames before the first heavy processing block
        fusion_metrics = self.fusion.get_display_metrics()
        fusion_metrics.update({
            "lidar_p_surface"     : 0.0,
            "lidar_p_smoke"       : 0.0,
            "lidar_spatial"       : 0.0,
            "lidar_valid_echo"    : 0.0,
            "lidar_beta"          : 0.0,
            "lidar_obstacle_prox" : 0.0,
        })

        print("[Pipeline] Running... Press Q to quit")
        print("-" * 60)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame = cv2.resize(frame, (640, 480))

            curr_t = time.time()
            fps    = 1.0 / (curr_t - prev_t + 1e-8)
            prev_t = curr_t

            # ── Live thermal feed (smooth, every frame) ──────────────
            live_active, _, _ = self.sensor_mgr.get_active_frame(frame, 0.25)

            # ── Heavy block every PROCESS_EVERY frames ───────────────
            if cached is None or frame_count % PROCESS_EVERY == 0:

                active_frame, mode, sensor = \
                    self.sensor_mgr.get_active_frame(frame, 0.25)

                # Gabor coherence
                gray = cv2.cvtColor(active_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 240)).astype(np.float32) / 255.0
                coh_result = self.coherence.compute(gray)

                # Depth
                depth = self.depth_est.predict(active_frame)

                depth_norm = (depth - depth.min()) / \
                             (depth.max() - depth.min() + 1e-8)
                dw = depth_norm.shape[1]

                def depth_region_score(region):
                    closeness = float(np.mean(region))
                    openness  = 1.0 - closeness
                    return float(np.clip(0.1 + 0.9 * openness, 0.1, 1.0))

                coh_result["left_conf"]   = depth_region_score(
                    depth_norm[:, :dw//3])
                coh_result["center_conf"] = depth_region_score(
                    depth_norm[:, dw//3 : 2*dw//3])
                coh_result["right_conf"]  = depth_region_score(
                    depth_norm[:, 2*dw//3:])

                # YOLO detection
                detections    = self.detector.detect(active_frame)
                dynamic_flags = []
                for det in detections:
                    is_moving, _ = self.detector.is_dynamic(
                        active_frame, det["box"])
                    dynamic_flags.append(is_moving)

                # ── Build GaborPipelineOutput from coh_result ─────────
                gabor_out = self._build_gabor_output(coh_result)

                # ── Run LiDAR pipeline (placeholder pulses for now) ───
                pulses       = self._generate_placeholder_pulses(gabor_out.global_confidence)
                lidar_result = self.lidar_pipeline.process_scan(pulses)
                bridge       = to_fusion_output(lidar_result)

                lidar_out = LidarPipelineOutput(
                    p_surface            = bridge.p_surface,
                    p_smoke              = bridge.p_smoke,
                    p_unknown            = bridge.p_unknown,
                    spatial_consistency  = bridge.spatial_consistency,
                    valid_echo_fraction  = bridge.valid_echo_fraction,
                    beta_estimate        = bridge.beta_estimate,
                    obstacle_proximity   = bridge.obstacle_proximity,
                )

                # ── Fusion update (real Gabor + real LiDAR output) ───
                ups = self.fusion.update(
                    gabor      = gabor_out,
                    lidar      = lidar_out,
                    odometry   = OdometryInput(),   # stationary for now
                    radar_flag = 0,
                )

                # Pull display metrics once per heavy block
                fusion_metrics = self.fusion.get_display_metrics()
                fusion_metrics.update({
                    "lidar_p_surface"     : bridge.p_surface,
                    "lidar_p_smoke"       : bridge.p_smoke,
                    "lidar_spatial"       : bridge.spatial_consistency,
                    "lidar_valid_echo"    : bridge.valid_echo_fraction,
                    "lidar_beta"          : bridge.beta_estimate,
                    "lidar_obstacle_prox" : bridge.obstacle_proximity,
                })

                cached = dict(
                    coh_result    = coh_result,
                    depth         = depth,
                    detections    = detections,
                    dynamic_flags = dynamic_flags,
                )

            # ── Unpack cache ─────────────────────────────────────────
            coh_result    = cached["coh_result"]
            depth         = cached["depth"]
            detections    = cached["detections"]
            dynamic_flags = cached["dynamic_flags"]
            mean_conf     = coh_result["global_conf"]

            # ── Draw on LIVE frame ───────────────────────────────────
            frame_det = self.detector.draw(
                live_active.copy(), detections, dynamic_flags
            )

            tier, color = self.get_tier(mean_conf)

            # Top overlay now shows fusion nav_state in addition to tier
            nav_state  = fusion_metrics.get("nav_state", "BLIND")
            nav_color  = self._nav_state_color(nav_state)

            cv2.putText(frame_det,
                f"Sensor:THERMAL | Conf:{mean_conf:.3f} | Tier:{tier}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 2)
            cv2.putText(frame_det,
                f"NavState:{nav_state}  "
                f"Objs:{len(detections)}  "
                f"Dyn:{sum(dynamic_flags)}  "
                f"FPS:{fps:.0f}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, nav_color, 1)
            cv2.putText(frame_det,
                "THERMAL CAMERA",
                (10, 470), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (200, 200, 200), 1)

            # ── Depth map ─────────────────────────────────────────────
            d_vis = cv2.normalize(
                depth, None, 0, 255, cv2.NORM_MINMAX
            ).astype('uint8')
            d_col = cv2.applyColorMap(d_vis, cv2.COLORMAP_MAGMA)
            d_col = cv2.resize(d_col, (640, 480))
            cv2.putText(d_col,
                "DEPTH MAP — Depth Anything V2",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)

            # ── Phase coherence map ───────────────────────────────────
            conf_map_resized = cv2.resize(
                coh_result["conf_map"], (640, 480)
            )
            c_vis = (conf_map_resized * 255).astype('uint8')
            c_col = cv2.applyColorMap(c_vis, cv2.COLORMAP_VIRIDIS)
            cv2.putText(c_col,
                "PHASE COHERENCE MAP",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
            cv2.putText(c_col,
                f"L:{coh_result['left_conf']:.2f} "
                f"C:{coh_result['center_conf']:.2f} "
                f"R:{coh_result['right_conf']:.2f}",
                (10, 465), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

            # ── Metrics panel ─────────────────────────────────────────
            metrics = self.metrics_panel(
                coh_result, tier, color,
                detections, dynamic_flags,
                fps,
                fusion_metrics,
            )

            # ── Compose 2×2 grid ──────────────────────────────────────
            frame_det = cv2.resize(frame_det, (640, 320))
            d_col     = cv2.resize(d_col,     (640, 320))
            c_col     = cv2.resize(c_col,     (640, 320))
            metrics   = cv2.resize(metrics,   (640, 320))

            top     = np.hstack([frame_det, d_col])
            bottom  = np.hstack([c_col, metrics])
            display = np.vstack([top, bottom])

            cv2.namedWindow("UAV Perception Pipeline", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("UAV Perception Pipeline", 1280, 680)
            cv2.imshow("UAV Perception Pipeline", display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    pipeline = UAVPipeline()
    pipeline.run()