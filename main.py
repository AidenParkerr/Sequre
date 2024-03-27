from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from utils.config_reader import ConfigReader
from utils.roi_initialiser import ROIInitialiser
from utils.video_utils import open_video
from utils.visualise import display, draw_grid, retrieve_cropped_box
from yolo.detector import YOLODetector

output_path = Path('output')
output_path.mkdir(exist_ok=True)
OUTPUT_DIR = output_path.resolve()


def handle_event(frame, box_class):
  if not box_class == 'person' or box_class == 'car':
    return
  timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  save_path = (OUTPUT_DIR / f"{box_class}-{timestamp}.jpg").as_posix()
  cv2.imwrite(save_path, frame)
  print(f"{timestamp}: `{box_class}` detected in the region of interest")

def compute_grid_coords(left: int, top: int, right: int, bottom: int) -> tuple[np.ndarray, np.ndarray]:
  grid_x, grid_y = np.meshgrid(np.linspace(left, right, num=10), np.linspace(top, bottom, num=10))
  return grid_x, grid_y

def obj_in_roi(box_coords, roi_points, percentage_threshold=0.3):
  """
  Check if the object is inside the region of interest.
  
  This function checks if the object is inside the region of interest by creating 
  a grid of points inside the bounding box and checking if the points are inside the 
  polygon to some threshold. The threshold is set to 50% by default.
  
  `grid_x` and `grid_y` are the grid points inside the bounding box.
  grid_points is a transpose of the grid points, this is done to change the shape of 
  the array from (2, 100) to (100, 2). This allows us to iterate over each point in the grid.

  Parameters
  ----------
      box_coords (list): The bounding box coordinates of the object.
      roi_points (list): The region of interest points.
      percentage_threshold (float, optional): The threshold for the percentage of points inside the polygon. 
      Defaults to 0.3 (30%).

  Returns
  -------
      bool: True if the object is inside the region of interest, False otherwise.
      
  Notes
  -----
      - The `cv2.pointPolygonTest` function is used to check if a point is inside the polygon.
      - The function returns a positive value if the point is inside the polygon, 0 if the point is on the edge of the polygon, 
        and a negative value if the point is outside the polygon.
  """
  left, top, right, bottom = np.array(box_coords, dtype=int).squeeze()
  
  # Create a grid of points inside the bounding box with 10x10 points.
  grid_x, grid_y = compute_grid_coords(left, top, right, bottom)
  grid_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T

  # Check each point in the grid.
  inside_points = sum([cv2.pointPolygonTest(roi_points, tuple(point), False) >= 0 for point in grid_points])

  # Calculate the percentage of points inside the polygon.
  total_points = grid_points.shape[0]
  inside_percentage = (inside_points / total_points)
  print(f"inside_points: {inside_points}, total_points: {total_points}")
  print(inside_percentage)
  return inside_percentage >= percentage_threshold


def init_roi(capture: cv2.VideoCapture) -> list:
  roi_initialiser = ROIInitialiser()
  ret, frame = capture.read()
  if not ret:
    raise ValueError("Error reading video")
  
  roi_points = roi_initialiser.initialise_roi(frame)
  return roi_points


# Main entry point of the application
def main(video_path: str, model_path: str):
  detector = YOLODetector(model_path)
  capture = open_video(video_path) # Open the video file and return the capture object
  roi_points = init_roi(capture) # Initialise the region of interest
  model_classes = detector.model.names
  
  while capture.isOpened():
    ret, frame = capture.read() # Read the frame from the video
    if not ret:
      break
  
    if len(roi_points) >= 2:
      cv2.polylines(frame, [np.array(roi_points)], False, (0, 255, 0), 2)
    
    results = detector.detect(frame) # Detect objects
    
    for box in results[0].boxes:
      box_class = model_classes[int(box.cls)] # Get the class of the object
      box_coords = box.xyxy.cpu().squeeze()
      # cropped_frame = retrieve_cropped_box(frame, box_coords, box_class, desired_class='person')
      draw_grid(frame, *compute_grid_coords(*box_coords), box_coords)
      if obj_in_roi(box_coords, roi_points):
        handle_event(results[0].plot(), box_class)
    
    if not display(results[0].plot()):
      break
    
  capture.release()
  cv2.destroyAllWindows()
  
  
if __name__ == '__main__':
  config_reader = ConfigReader('configs/config.json')

  video_path = 'data/raw/sample_video.mp4'
  model_path = config_reader.get('model_config')['model_path']
  
  main(video_path, model_path)