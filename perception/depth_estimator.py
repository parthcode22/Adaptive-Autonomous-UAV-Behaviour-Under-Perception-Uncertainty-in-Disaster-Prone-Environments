import torch
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, model_type="MiDaS_small"):
        print('[DepthEstimator]loading MiDaS model...')
        self.model=torch.hub.load("intel-isl/MiDaS",model_type)
        self.model.eval()
        transforms=torch.hub.load("intel-isl/MiDas", "transforms")
        self.transform =transforms.small_transform
        print("[DepthEstimator]Model ready")

    def predict(self, bgr_image):
        rgb=cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        inp=self.transform(rgb)
        with torch.no_grad():
            depth =self.model(inp).squeeze().numpy()
        depth =cv2.resize(depth,(bgr_image.shape[1],bgr_image.shape[0]))
        return depth
    

