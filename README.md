# Adaptive-Autonomous-UAV-Behaviour-Under-Perception-Uncertainty-in-Disaster-Prone-Environments

## Overview

This project focuses on developing a confidence-aware perception framework for autonomous UAV navigation in disaster-prone environments.

Traditional UAV systems rely heavily on RGB cameras and continue navigating even when perception quality deteriorates due to smoke, dust, poor lighting, motion blur, or environmental uncertainty.

This project introduces a confidence estimation module that continuously evaluates the reliability of perception data and adaptively switches sensing modalities to improve robustness and safety.

---

## Key Features

* Monocular Depth Estimation using MiDaS
* Confidence Estimation for perception reliability
* Adaptive Sensor Switching

  * RGB Mode
  * IR Mode (simulated)
  * Thermal Mode (simulated)
* Real-time Visualization
* Disaster-Oriented Navigation Framework
* Future DRL Integration

---

## System Architecture

RGB Camera
↓
Depth Estimation (MiDaS)
↓
Confidence Estimation
↓
Adaptive Sensor Selection
(RGB → IR → Thermal)
↓
Navigation Decision Module
↓
Ground Station / Telemetry

---

## Project Structure

```text
uav_navigation/
│
├── perception/
│   ├── __init__.py
│   ├── depth_estimator.py
│   ├── confidence_estimator.py
│   └── sensor_manager.py
│
├── tests/
│
├── run_pipeline.py
│
├── requirements.txt
│
└── README.md
```

## Implemented Modules

### 1. Depth Estimator

Uses MiDaS Small model for monocular depth estimation.

Input:

* RGB Frame

Output:

* Relative Depth Map

---

### 2. Confidence Estimator

Computes confidence based on:

* Depth Consistency
* Image Sharpness
* Brightness Quality
* Edge Density

Output:

* Confidence Score (0–1)

---

### 3. Sensor Manager

Adaptive sensor switching logic:

| Confidence  | Sensor  |
| ----------- | ------- |
| > 0.75      | RGB     |
| 0.45 – 0.75 | IR      |
| < 0.45      | Thermal |

---

## Research Novelty

Most existing UAV navigation systems:

* Use RGB-only perception.
* Ignore perception uncertainty.
* Make navigation decisions regardless of confidence.

This work introduces:

* Confidence-aware perception.
* Adaptive sensor switching.
* Uncertainty-guided autonomous behaviour.
* Disaster-specific robustness improvements.

---

## Current Progress

Completed:

* Depth Estimation Module
* Confidence Estimation Module
* Sensor Manager Module
* Real-Time Visualization Pipeline

In Progress:

* Confidence Calibration
* AirSim Integration

Upcoming:

* PPO-Based DRL Navigation
* Confidence-Aware Reward Function
* Dynamic Obstacle Avoidance
* Victim Detection
* Multi-UAV Coordination

---

## Installation

```bash
conda create -n drone_model python=3.10
conda activate drone_model

pip install torch torchvision
pip install opencv-python numpy matplotlib
pip install timm einops
```

## Run

```bash
python run_pipeline.py
```

---

## Expected Outcomes

* Reliable depth perception
* Confidence-aware navigation
* Adaptive sensing strategy
* Improved disaster response capability
* Research publication potential

---

## Author

Parth Vinod Indulkar

B.Tech Electronics and Computer Science Engineering

Research Intern – AI Excellence Lab

Mumbai, India
