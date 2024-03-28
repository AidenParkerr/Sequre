import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from data.data_handler import DataHandler
from utils.config_reader import ConfigReader
from utils.roi_initialiser import ROIInitialiser
from utils.visualise import display, draw_grid, retrieve_cropped_box
from yolo.detector import YOLODetector


class Sequre:
  def __init__(self, config_path: str):
    self.video_config, \
        self.model_path, \
        self.roi_points = self._init_dependencies(config_path)
    self.video_name = Path(self.video_config['video_path']).stem
    self.data_handler = DataHandler(self.video_config)

    FOLDER_TIMESTAMP = datetime.now().strftime("%Y-%m-%d")
    output_path = Path(
        'data/output/').joinpath(FOLDER_TIMESTAMP, self.video_name)
    output_path.mkdir(parents=True, exist_ok=True)
    self.output_dir = output_path.resolve()

  def _init_dependencies(
          self, config_path) -> tuple[dict, str, np.ndarray]:
    """
    Initialise the dependencies for the application.

    Handle the initialisation of the dependencies for the application. This function reads the
    configuration file, extracts the video path, video dimensions, model path, and region of interest
    points from the configuration file.

    Parameters
    ----------
    config_path : str
        The path to the configuration file.

    Returns
    -------
    tuple[dict, str, np.ndarray]sample_video
        A tuple containing the video configuration, sample_videomodel path, and region of interest points.

    """
    config_reader = ConfigReader(config_path)
    video_config = config_reader.get('video_config')

    # load the roi points for the selected video file from the config file, if
    # available
    roi_points_dir = video_config['roi_points_dir']
    roi_points_path = Path(roi_points_dir).joinpath(
        f"{Path(video_config['video_path']).stem}_roi_points.npy")
    roi_points = np.load(roi_points_path) if Path(
        roi_points_path).exists() else np.array([], dtype=np.int32)
    model_path: str = config_reader.get('model_config')['model_path']
    return video_config, model_path, roi_points

  def handle_object_in_roi(self, frame, object_class,
                           classes_to_monitor=['person', 'car']):
    """
    Handle the object inside the region of interest.

    This function saves the frame to the output directory if the object is inside the region of interest.

    Parameters
    ----------
    frame : np.ndarray
        The frame containing the object.
    box_class : str
        The class of the object.
    """
    if object_class not in classes_to_monitor:
      return

    def save_frame():
      """
      Save the frame to the output directory.
      """
      timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      save_path = (
          self.output_dir /
          f"{object_class}-{timestamp}.jpg").as_posix()
      cv2.imwrite(save_path, frame)

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
    left : int
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
    return grid_x, grid_y

  def is_object_in_roi(self, box_coords: np.ndarray, roi_points: np.ndarray,
                       percentage_threshold: float=0.25):
    """
    Check if the object is inside the region of interest.

    This function checks if the object is inside the region of interest by creating
    a grid of points inside the bounding box and checking if the points are inside the
    polygon to some threshold. The threshold is set to 25% by default.

    `grid_x` and `grid_y` are the grid points inside the bounding box.
    grid_points is a transpose of the grid points, this is done to change the shape of
    the array from (2, 100) to (100, 2). This allows us to iterate over each point in the grid.
stairwell
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
    grid_points = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    # Check each point in the grid.
    inside_points = sum([cv2.pointPolygonTest(
        np.array(roi_points), tuple(point), False) >= 0 for point in grid_points])

    # Calculate the percentage of points inside the polygon.
    total_points = grid_points.shape[0]
    inside_percentage = (inside_points / total_points)
    return inside_percentage >= percentage_threshold

  def init_roi(self, capture: cv2.VideoCapture) -> np.ndarray:
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
        video_name=self.video_name,
        roi_points=self.roi_points)

    success, frame = capture.read()
    if not success:
      print("Error reading the video file")
      capture.release()
      cv2.destroyAllWindows()
      return np.array([])

    frame = self.data_handler.process_frame(frame)
    roi_points = roi_initialiser.set_roi(frame)
    return roi_points

  # Main entry point of the application
  def main(self, debug=False):
    # Open the video file and return the capture object
    detector = YOLODetector(self.model_path)
    capture = self.data_handler.get_capture()
    model_classes = detector.model.names

    if not len(self.roi_points) > 0:
      self.roi_points = self.init_roi(capture)

    while capture.isOpened():
      success, frame = capture.read()  # Read the frame from the video
      if not success:
        break

      frame = self.data_handler.process_frame(frame)  # Process the frame

      results = detector.detect(frame)  # Detect objects

      # Loop through the detected objects and check if they are inside the
      # region of interest
      for box in results[0].boxes:
        box_class = model_classes[int(box.cls)]  # Get the class of the object
        box_coords = box.xyxy.cpu().squeeze()  # Get the bounding box coordinates
        if debug:
          draw_grid(frame, *self.compute_grid_coords(*box_coords), box_coords)
        # cropped_frame = retrieve_cropped_box(frame, box_coords, box_class, desired_class='person')
        if self.is_object_in_roi(box_coords, self.roi_points) and len(
                self.roi_points) > 0:
          self.handle_object_in_roi(frame, 
                                    object_class=box_class, 
                                    classes_to_monitor=[
                                      'person',
                                      'car'])

      if debug:
        # Draw the region of interest on the frame and display the frame with
        # bounding boxes
        frame = results[0].plot()
        cv2.polylines(
            frame, [
                self.roi_points], isClosed=True, color=(
                0, 255, 0), thickness=2)

      if not display(frame):
        break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
  sequre = Sequre('configs/config.json')
  sequre.main()
