import cv2
import numpy as np
import logging

class VideoHandler:
  """
  A class that handles video data processing.

  Parameters
  ----------
  video_path : str
    The path to the video file.
  video_dim : tuple[int, int]
    The desired dimensions of the video frames.
  """

  def __init__(self, video_config: dict):
    self.logger = logging.getLogger(__name__)
    self.video_path, self.video_dim = self._init_video_config(video_config)

  def _init_video_config(
          self, video_config: dict[str, str]) -> tuple[str, tuple[int, int]]:
    """
    Initialise the video data.

    This function initialises the video data by extracting the video path, video dimensions,
    and region of interest points from the configuration file. If the region of interest points
    file is not available, an empty numpy array is returned, this will prompt the user to select
    the region of interest points manually.

    Args:
        video_config (dict[str, str]): The video configuration dictionary.

    Returns:
        tuple[str, tuple[int, int]]: A tuple containing the video path
        and video dimensions.
    """
    video_path: str = video_config['video_path']
    self.logger.info(f"Video path: {video_path}")
    # parse the video dimensions from the config file
    raw_video_dims: str = video_config.get('video_dimensions', '1280x720')
    width, height = (int(dim) for dim in raw_video_dims.split('x'))
    video_dims: tuple[int, int] = (width, height)
    self.logger.info(f"Video dimensions: {video_dims}")
    return video_path, video_dims

  def get_resolution(self, capture: cv2.VideoCapture) -> tuple[int, int]:
    """
    Get the resolution of the video capture.

    Parameters
    ----------
    capture : cv2.VideoCapture
      The video capture object.

    Returns
    -------
    tuple[int, int]
      The resolution of the video capture.
    """
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return width, height
  
  def get_capture_fps(self, capture: cv2.VideoCapture) -> int:
    """
    Get the frames per second of the video capture.

    Parameters
    ----------
    capture : cv2.VideoCapture
      The video capture object.

    Returns
    -------
    int
      The frames per second of the video capture.
    """
    return int(capture.get(cv2.CAP_PROP_FPS))

  def get_capture(self) -> cv2.VideoCapture:
    """
    Open the video file and return the capture object.

    Raises
    ------
    ValueError
      If the video file does not exist or if there is an error opening the video.

    Returns
    -------
    cv2.VideoCapture
      The video capture object.
    """

    self.logger.info("Opening the video file..")
    capture = cv2.VideoCapture(self.video_path)
    if not capture.isOpened():
      raise ValueError("Error opening video")
    return capture

  def process_frame(self, frame: np.ndarray) -> np.ndarray:
    """
    Flexible function for multi-point processing of the frame. Currently processes the 
    frame by resizing it to the desired dimensions.

    Parameters
    ----------
    frame : np.ndarray
      The frame to process.

    Returns
    -------
    np.ndarray
      The processed frame.
    """
    resized_capture = self.resize_frame(frame)
    return resized_capture

  def resize_frame(self, frame: np.ndarray) -> np.ndarray:
    """
    Resize the frame to the desired dimensions.

    Parameters
    ----------
    frame : np.ndarray
      The frame to resize.

    Returns
    -------
    np.ndarray
      The resized frame.
    """
    return cv2.resize(frame, self.video_dim, interpolation=cv2.INTER_AREA)
