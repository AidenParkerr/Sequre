import os

import cv2
import numpy as np


class DataHandler:
  """
  A class that handles video data processing.

  Parameters
  ----------
  video_path : str
    The path to the video file.
  video_dim : tuple[int, int]
    The desired dimensions of the video frames.
  """

  def __init__(self, video_path: str, video_dim: tuple[int, int]):
    self.video_path: str = video_path
    self.video_dim: tuple = video_dim

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
    if not os.path.exists(self.video_path):
      raise ValueError("Video file does not exist")

    capture = cv2.VideoCapture(self.video_path)
    if not capture.isOpened():
      raise ValueError("Error opening video")
    return capture

  def process_frame(self, frame: np.ndarray) -> np.ndarray:
    """
    Processes the frame by resizing it to the desired dimensions.

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
