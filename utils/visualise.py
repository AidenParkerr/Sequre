import cv2
import numpy as np


def display(frame, window_name='frame') -> bool:
  cv2.imshow(window_name, frame)
  if cv2.waitKey(25) & 0xFF == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
    return False
  return True

  
def draw_grid(frame, grid_x, grid_y, box_coords) -> None:
  left, top, right, bottom = np.array(box_coords, dtype=int).squeeze()
  for i in range(10):
    cv2.line(frame, (int(grid_x[0][i]), top), (int(grid_x[0][i]), bottom), (0, 0, 255), 2)
    cv2.line(frame, (left, int(grid_y[i][0])), (right, int(grid_y[i][0])), (0, 0, 255), 2)
  

def retrieve_cropped_box(frame, box_coords, box_class, desired_class='person') -> np.ndarray:
  if box_class != desired_class:
    return np.array([])
  left, top, right, bottom = np.array(box_coords, dtype=int).squeeze()
  cropped_frame = frame[top:bottom, left:right]
  return cropped_frame
  