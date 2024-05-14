import hashlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config.config_reader import ConfigReader
from utils.file_utils import load_roi_points
from utils.logger import setup_logging
from utils.roi_initialiser import ROIInitialiser
from utils.video_utils.video_handler import VideoHandler
from utils.visualise import display, draw_grid
from yolo.detector import YOLODetector


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
  def __init__(self, config_path: str = 'config/config.json'):
    self.TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H-%M-%S") # Get the current timestamp
    
    self.logger = setup_logging()
    self.app_config = ConfigReader(config_path).config
    self.video_handler = VideoHandler(self.app_config['video_config'])
    
    # Create a hash of the video path to uniquely identify the video file and output
    video_path = str(self.video_handler.video_path).lower()
    self.video_hash = hashlib.sha256(video_path.encode()).hexdigest()
    self.output_dir = self._setup_output_directory()


  def _setup_output_directory(self):
    """ Setup the output directory for saving the frames. """
    timestamp = self.TIMESTAMP.split(' ')[0] # Get the date part of the timestamp
    output_path = Path('data/output/').joinpath(timestamp, self.video_hash)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


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
    self.logger.info(
        f"'{object_class}' detected inside the region of interest")
    # Spin new thread for saving the frame
    # threading.Thread(target=self.handle_object_in_roi).start()
    if object_class not in classes_to_monitor:
      self.logger.info(f"Object class '{object_class}' not in classes to monitor")
      return 
    
    timestamp = datetime.now().strftime("%H-%M-%S")
    image_name = f"{object_class}-{timestamp}.jpg"
    save_path = self.output_dir / image_name
    cv2.imwrite(str(save_path.resolve()), frame)
    self.logger.info(f"Frame with object '{object_class}' saved to '{save_path}'")

  def compute_grid_coords(self, left: int, top: int, right: int,
                          bottom: int, h_num: int=10, v_num: int=10) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the grid coordinates inside the bounding box.

    This function computes the grid coordinates inside the bounding box by creating a meshgrid
    of points inside the bounding box.
    The grid points are created using the `np.linspace` function to create 10x10 points inside the
    bounding box. The `grid_x` and `grid_y` are the meshgrid points inside the bounding box.

    Parameters
    ----------
    left : int
      The left coordinate of the bounding box.
    top : int
      The top coordinate of the bounding box.
    right : int
      The right coordinate of the bounding box.
    bottom : int
      The bottom coordinate of the bounding box.
    h_num : int
      The number of horizontal points in the grid. Defaults to 10.
    v_num : int
      The number of vertical points in the grid. Defaults to 10.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
      The grid coordinates inside the bounding box.
    """
    grid_x, grid_y = np.meshgrid(
        np.linspace(
            left, right, num=h_num), np.linspace(
            top, bottom, num=v_num))
    self.logger.info(f"Grid coordinates of bounding box computed successfully")
    return grid_x, grid_y

  def is_object_in_roi(self, box_coords: np.ndarray, roi_points: np.ndarray,
                       percentage_threshold: float = 0.25):
    """
    Check if the object is inside the region of interest.

    This function checks if the object is inside the region of interest by creating
    a grid of points inside the bounding box and checking if the points are inside the
    polygon to some threshold. The threshold is set to 10% by default.

    `grid_x` and `grid_y` are the grid points inside the bounding box.
    grid_points is a transpose of the grid points, this is done to change the shape of
    the array from (2, 100) to (100, 2). This allows us to iterate over each point in the grid.
    
    
    Parameters
    ----------
        box_coords (list): The bounding box coordinates of the object.
        roi_points (list): The region of interest pooffice_robbersints.
        percentage_threshold (float, optional): The threshold for the percentage of points inside the polygon.
        Defaults to 0.25 (25%).

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
    
    # Transpose the grid points to change the shape of the array from (2, 100) to (100, 2).
    grid_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    # Check each point in the grid. #? Could be optimized
    inside_points = sum([cv2.pointPolygonTest(
        np.array(roi_points), tuple(point), False) >= 0 for point in grid_points])

    # Calculate the percentage of points inside the polygon.
    total_points = grid_points.shape[0]
    inside_percentage = (inside_points / total_points)
    self.logger.info(
        f"Percentage of points inside the ROI: {inside_percentage * 100:.2f}%")
    return inside_percentage >= percentage_threshold

  def init_roi(self, roi_points: np.ndarray) -> np.ndarray:
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
        video_name=self.video_hash,
        roi_points=roi_points)

    capture = self.video_handler.get_capture()
    success, frame = capture.read()
    if not success:
      self.logger.error("Error reading the video file")
      capture.release()
      cv2.destroyAllWindows()
      return np.array([])

    frame = self.video_handler.process_frame(frame)
    self.logger.info("Setting region of interest points..")
    roi_points = roi_initialiser.get_roi(frame)
    self.logger.info(f"Region of interest points set successfully")
    return roi_points

  
  def get_roi_points(self) -> np.ndarray:
    """ Get the region of interest points from the file. If the file is not available, 
        prompt user to initialise the ROI points."""
    file_name = f"{self.video_hash}_roi_points.npy"
    roi_points_path = Path(self.app_config['video_config']['roi_points_dir']) / file_name
    roi_points = load_roi_points(str(roi_points_path)) # Load the ROI points from the file
    return roi_points 

  def _debug(self, frame: np.ndarray, box_coords: np.ndarray, roi_points: np.ndarray) -> np.ndarray:
    # Draw the region of interest on the frame and display the frame with
    # bounding boxes
    cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)
    self.logger.info(f"Bounding box coordinates: {box_coords}")
    grid_coords = self.compute_grid_coords(*box_coords)
    draw_grid(frame, *grid_coords, box_coords)
    return frame
    
    
  # Main entry point of the application
  def main(self, debug=False):
    detector = YOLODetector(self.app_config['model_config'])
    capture = self.video_handler.get_capture()
    roi_points = self.get_roi_points() 
    if roi_points.size == 0:
      roi_points = self.init_roi(roi_points)
    while capture.isOpened():
      _, frame = capture.read()
      assert frame is not None, "Frame is None"

      orig_frame = self.video_handler.process_frame(frame)
      results = detector.predict(orig_frame)            
      
      for box in results[0].boxes: # Iterate over the detected objects
        box_class = detector.model.names[int(box.cls)]
        # Coordinates in the format (left, top, right, bottom)
        box_coords = box.xyxy.cpu().squeeze()
        
        if debug: # Display the frame with bounding boxes and region of interest
          frame = frame.copy()
          frame = self._debug(orig_frame, box_coords, roi_points)

        if self.is_object_in_roi(box_coords, roi_points):
          self.handle_object_in_roi(orig_frame, object_class=box_class)

      if not display(orig_frame, delay=1):
        self.logger.info("Exiting the video processing..")
        break
      
    capture.release()
    cv2.destroyAllWindows()
    self.logger.info("Video processing completed")




if __name__ == '__main__':
  sequre = Sequre()
  sequre.main()
