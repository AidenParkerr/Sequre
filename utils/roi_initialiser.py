import cv2
import numpy as np


class ROIInitialiser:
  def __init__(self) -> None:
    self.roi_points = []
    self.done = False
    self.current = (0, 0)
    self.frame = None
    
  def _on_mouse(self, event, x, y, flags, param):
    if self.done:
      return
    
    if event == cv2.EVENT_MOUSEMOVE:
      self.current = (x, y)
    elif event == cv2.EVENT_LBUTTONDOWN:
      self.roi_points.append((x, y))
    elif event == cv2.EVENT_MBUTTONDOWN:
      if len(self.roi_points) > 1:
        self.done = True
    elif event == cv2.EVENT_RBUTTONDOWN:
      self.roi_points = []


    if not self.done:
      updated_frame = self.frame.copy()
      if len(self.roi_points) > 0:
        cv2.polylines(updated_frame, [np.array(self.roi_points)], False, (0, 255, 0), 2)
        cv2.line(updated_frame, self.roi_points[-1], self.current, (0, 255, 0), 2)
      cv2.imshow('frame', updated_frame)      
        
  def initialise_roi(self, frame) -> np.ndarray:
    self.frame = frame
    cv2.putText(frame, "Select ROI points. Press middle mouse button to finish.", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('frame', frame)
    cv2.setMouseCallback('frame', self._on_mouse)
    
    while not self.done:
      cv2.waitKey(25)
      if cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        break
    cv2.setMouseCallback('frame', lambda *args: None)
    
    return np.array(self.roi_points, dtype=int)
