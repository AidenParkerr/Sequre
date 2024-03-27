import cv2
from ultralytics import YOLO

class YOLODetector:
  def __init__(self, model_path) -> None:
    self.model = YOLO(model_path)
    self.model.fuse()

  def detect(self, frame) -> list:
    results = self.model(frame)
    return results

