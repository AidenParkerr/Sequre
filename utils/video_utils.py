import cv2

def open_video(video_path: str) -> cv2.VideoCapture:
  capture = cv2.VideoCapture(video_path)
  if not capture.isOpened():
    raise ValueError("Error opening video")
  return capture