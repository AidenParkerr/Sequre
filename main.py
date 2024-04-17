import hashlib
import logging
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from data.data_handler import DataHandler
from utils.config_reader import ConfigReader
from utils.file_utils import load_roi_points
from utils.logger import setup_logging
from utils.roi_initialiser import ROIInitialiser
from utils.visualise import display, draw_grid, retrieve_cropped_box
from yolo.detector import YOLODetector
import torch

class Sequre:
  """
  The main application class for the object detection application.
  
  This application allows the user to detect objects in a video file and save the frames
  containing the objects inside the region of interest. The region of interest is defined by the
  user by selecting the points on the frame using the mouse. The application uses the YOLO object
  detector to detect objects in the video file.
  
  Parameters
  ----------
  config_path : str
    The path to the configuration file.
    
  Attributes
  ----------
  video_config : dict
    The video configuration settings.
  model_path : str
    The path to the model file.
  data_handler : DataHandler
    The data handler object for the application.
  video_path_hash : str
    The hash of the video path.
  output_dir : Path
    The output directory for the application.
  logger : logging.Logger
    The logger object for the application.   
    
  #? TODO: 
  - Add advanced frame handling for laggier videos and live streams.
  - Add support for multiple video files.
  - Add support for multiple region of interest points.
  - Add support for multiple object classes to monitor. #* Done
  """
  def __init__(self, config_path: str):
    setup_logging()
    self.logger = logging.getLogger('ObjectDetectionLogger')
    self.logger.info("Initialising the application")
    self.video_config: dict = ConfigReader(config_path).get('video_config')
    self.model_path: str = ConfigReader(config_path).get('model_config')['model_path']
    self.data_handler = DataHandler(self.video_config)
    self.video_path_hash = self._generate_hash(str(self.data_handler.video_path))
    
    FOLDER_TIMESTAMP = self._create_timestamp("%Y-%m-%d")
    output_path = Path(
        'data/output/').joinpath(FOLDER_TIMESTAMP, self.video_path_hash)
    output_path.mkdir(parents=True, exist_ok=True)
    self.output_dir = output_path.resolve()
    self.logger.info(f"Output directory: '{self.output_dir}'")


  def _generate_hash(self, url: str) -> str:
    """
    Generate a hash from the URL.
    
    This function generates a hash from the URL using the SHA256 algorithm.

    Parameters
    ----------
    url : str
      The URL to generate the hash from.
        
    Returns
    -------
    str
      The hash generated from the URL.
    """
    
    url_normal = url.lower()
    hash_object = hashlib.sha256(url_normal.encode())
    return hash_object.hexdigest()
    


  def _create_timestamp(self, format="%Y-%m-%d"):
    """
    Create a timestamp for the output directory.

    This function creates a timestamp for the output
    directory using the current date and time.

    Returns
    -------
    str
        The timestamp for the output directory.
    """
    return datetime.now().strftime(format)

  def handle_object_in_roi(self, frame, object_class,
                           classes_to_monitor=['person', 'car']):
    """
    Handle the object inside the region of interest.

    This function saves the frame to the output directory if the object
    is inside the region of interest.

    Parameters
    ----------
    frame : np.ndarray
        The frame containing the object.
    box_class : str
        The class of the object.
    """
    if object_class not in classes_to_monitor:
      self.logger.info(
          f"Detected object class '{object_class}' in ROI, though not in classes to monitor")
      return

    def save_frame():
      """
      Save the frame to the output directory.
      """
      timestamp = self._create_timestamp("%Y-%m-%d-%H-%M-%S")
      save_path = str((
          self.output_dir /
          f"{object_class}-{timestamp}.jpg").resolve())
      self.logger.info(f"Saving frame to: '{save_path}'")
      cv2.imwrite(save_path, frame)
      self.logger.info("Frame saved successfully")

    # Handle the saving of the frame in a separate thread
    threading.Thread(target=save_frame).start()

  def compute_grid_coords(self, left: int, top: int, right: int,
                          bottom: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the grid coordinates inside the bounding box.

    This function computes the grid coordinates inside the bounding box by creating a meshgrid
    of points inside the bounding box.
    The grid points are created using the `np.linspace` function to create 10x10 points inside the
    bounding box. The `grid_x` and `grid_y` are the meshgrid points inside the bounding box.

    Parameters
    ----------
    left : intFalse
      The left coordinate of the bounding box.
    top : int
      The top coordinate of the bounding box.
    right : int
      The right coordinate of the bounding box.
    bottom : int
      The bottom coordinate of the bounding box.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
      The grid coordinates inside the bounding box.
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(
            left, right, num=10), np.linspace(
            top, bottom, num=10))
    self.logger.info(f"Grid coordinates of bounding box computed successfully")
    return grid_x, grid_y

  def is_object_in_roi(self, box_coords: np.ndarray, roi_points: np.ndarray,
                       percentage_threshold: float = 0.1):
    """
    Check if the object is inside the region of interest.

    This function checks if the object is inside the region of interest by creating
    a grid of points inside the bounding box and checking if the points are inside the
    polygon to some threshold. The threshold is set to 10% by default.

    `grid_x` and `grid_y` are the grid points inside the bounding box.
    grid_points is a transpose of the grid points, this is done to change the shape of
    the array from (2, 100) to (100, 2). This allows us to iterate over each point in the grid.
stairwell
    Parameters
    ----------
        box_coords (list): The bounding box coordinates of the object.
        roi_points (list): The region of interest pooffice_robbersints.
        percentage_threshold (float, optional): The threshold for the percentage of points inside the polygon.
        Defaults to 0.10 (10%).

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
    grid_x, grid_y = self.compute_grid_coords(left, top, right, bottom)
    grid_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    # Check each point in the grid.
    inside_points = sum([cv2.pointPolygonTest(
        np.array(roi_points), tuple(point), False) >= 0 for point in grid_points])

    # Calculate the percentage of points inside the polygon.
    total_points = grid_points.shape[0]
    inside_percentage = (inside_points / total_points)
    self.logger.info(
        f"Percentage of points inside the ROI: {inside_percentage * 100:.2f}%")
    return inside_percentage >= percentage_threshold

  def init_roi(self, capture: cv2.VideoCapture, roi_points: np.ndarray) -> np.ndarray:
    """
    Initialise the region of interest.

    This function initialises the region of interest by setting the ROI points using the ROIInitialiser
    class. The ROI points are set by selecting the points on the frame using the mouse.

    Parameters
    ----------
    capture : cv2.VideoCapture
      The capture object for the video.

    Returns
    -------
    np.ndarray
      The region of interest points.
    """
    roi_initialiser = ROIInitialiser(
        video_name=self.video_path_hash,
        roi_points=roi_points)

    success, frame = capture.read()
    if not success:
      self.logger.error("Error reading the video file")
      capture.release()
      cv2.destroyAllWindows()
      return np.array([])

    frame = self.data_handler.process_frame(frame)
    self.logger.info("Setting region of interest points..")
    roi_points = roi_initialiser.set_roi(frame)
    self.logger.info(f"Region of interest points set successfully")
    return roi_points

  # Main entry point of the application
  def main(self, debug=False):
    # Open the video file and return the capture object
    detector = YOLODetector(self.model_path)
    print(torch.cuda.is_available())
    self.logger.info("Opening the video file..")
    capture = self.data_handler.get_capture()
    model_classes = detector.model.names
    roi_points_path = str(Path(self.video_config['roi_points_dir']).joinpath(
        f"{self.video_path_hash}_roi_points.npy"))
    roi_points = load_roi_points(roi_points_path)
    
    if not len(roi_points) > 0:
      roi_points = self.init_roi(capture, roi_points)

    while capture.isOpened():
      success, frame = capture.read()  # Read the frame from the video
      if not success:
        self.logger.error("Error reading the video file")
        break

      frame = self.data_handler.process_frame(frame)  # Process the frame
      print(self.data_handler.get_capture_fps(capture))
      results = detector.predict(frame)  # Detect objects
      if debug:
        # Draw the region of interest on the frame and display the frame with
        # bounding boxes
        frame = results[0].plot()
        cv2.polylines(
            frame, [
                roi_points], isClosed=True, color=(
                0, 255, 0), thickness=2)
        
      # Loop through the detected objects and check if they are inside the region of interest
      for box in results[0].boxes:
        box_class = model_classes[int(box.cls)]  # Get the class of the object
        box_conf = box.conf.cpu().item()
        self.logger.info(
            f"'{box_class}' detected in the frame with confidence: {box_conf:.2f}")
        # Get the bounding box coordinates in the format (left, top, right, bottom)
        box_coords = box.xyxy.cpu().squeeze()  
        if debug:
          self.logger.info(f"Bounding box coordinates: {box_coords}")
          grid_coords = self.compute_grid_coords(*box_coords)
          draw_grid(frame, *grid_coords, box_coords)
        # cropped_frame = retrieve_cropped_box(frame, box_coords, box_class, desired_class='person')
        if self.is_object_in_roi(box_coords, roi_points) and len(
                roi_points) > 0:
          self.logger.info(
              f"'{box_class}' detected inside the region of interest")
          self.handle_object_in_roi(frame,
                                    object_class=box_class,
                                    classes_to_monitor=[
                                        'person',
                                        'car'])

      if not display(frame, delay=1):
        self.logger.info("Exiting the video processing..")
        break

    capture.release()
    cv2.destroyAllWindows()
    self.logger.info("Video processing completed")


if __name__ == '__main__':
  sequre = Sequre('configs/config.json')
  sequre.main()
