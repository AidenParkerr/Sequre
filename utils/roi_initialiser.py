from pathlib import Path

import cv2
import numpy as np


class ROIInitialiser:
  """
  A class to initialise the region of interest in a video.

  The ROIInitialiser class provides a way to select the region of interest in a video.
  The user can select the region of interest by clicking on the frame to select the
  points that define the region of interest.
  The user can also clear the points by clicking the right mouse button and finish selecting
  the points by clicking the middle mouse button.

  The ROI points are saved to a numpy file for future use. The ROI points can be loaded from the
  numpy file to avoid selecting the points again.
  The ROI points are saved to the file with the name of the video file appended with '_roi_points.npy'.
  Example: `footage1_roi_points.npy`.

  The ROI points are saved to the in the following format:
    ```
    [[x1, y1],
     [x2, y2],
     [x3, y3],
       ...
     [xn, yn]]
    ```

  Parameters
  ----------
  video_name : str
    The name of the video file.
  roi_points : np.ndarray
    The region of interest points.

  Attributes
  ----------
  video_name : str
    The name of the video file.
  roi_points : np.ndarray
    The region of interest points.
  done : bool
    A flag to indicate if the user has finished selecting the region of interest.
  current : tuple
    The current point selected by the user.
  temp_points : list
    A list to store the temporary points selected by the user.

  Methods
  -------
  _save_roi_points(save_path: str) -> None
    Save the ROI points to a numpy file.
  _load_roi_points(load_path: str) -> np.ndarray
    Load the ROI points from a numpy file.
  handle_mouse_events(event, x, y, flags, param) -> None
    Handle mouse events for selecting ROI points in a frame.
  _handle_roi_setup(frame, window_name, params) -> None
    Handle the setup of the region of interest.
  set_roi(frame, window_name='Define Region of Interest') -> np.ndarray
    Set the region of interest in a frame.
  """

  def __init__(self, video_name: str, roi_points: np.ndarray) -> None:
    self.video_name: str = video_name
    self.roi_points = np.array(
        roi_points) if roi_points is not None else np.array([], dtype=np.int32)
    self.done: bool = False
    self.current: tuple = (0, 0)
    self.temp_points: list = []

  def _save_roi_points(self, save_path: str) -> None:
    """
    Save the ROI points to a numpy file.

    Parameters
    ----------
    save_path : str
      The path to save the ROI points to.
    """
    if not Path(save_path).parent.exists():
      Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(save_path, self.roi_points)

  def _load_roi_points(self, load_path: str) -> np.ndarray:
    """
    Load the ROI points from a numpy file.

    Parameters
    ----------
    load_path : str
      The path to load the ROI points from.
    """
    try:
      return np.load(load_path)
    except OSError as e:
      print(f"Input file does not exist or cannot be read: {e}")
      return np.array([])
    except EOFError as e:
      print(f"Calling np.load multiple times on the same file handle: {e}")
      return np.array([])
    except Exception as e:
      print(f"Error loading the region of interest points: {e}")
      return np.array([])

  def handle_mouse_events(self, event, x, y, flags, param) -> None:
      """
      Handle mouse events for selecting ROI points in a frame.

      Parameters
      ----------
      event : int
          The type of mouse event.
      x : int
          The x-coordinate of the mouse event.
      y : int
          The y-coordinate of the mouse event.
      flags : int
          The flags for the mouse event.
      param : dict
          A dictionary containing the ROI points and the current point.
      """
      if self.done:
          return

      if event == cv2.EVENT_MOUSEMOVE:
          # On mouse move, update the current point
          self.current = (x, y)
      elif event == cv2.EVENT_LBUTTONDOWN:
          if flags & cv2.EVENT_FLAG_SHIFTKEY:
              # On Shift + left mouse button click, finish selecting the ROI points if > 3 points
              if len(self.temp_points) > 3:
                  self.done = True
                  self.roi_points = np.array(self.temp_points, dtype=np.int32)
                  # Save the ROI points to a file
                  save_path = f'data/output/roi_points/{self.video_name}_roi_points.npy'
                  self._save_roi_points(save_path)
          else:
              # On left mouse button click, add the point to the ROI points
              self.temp_points.append((x, y))
      elif event == cv2.EVENT_RBUTTONDOWN:
          # On right mouse button click, clear the ROI points
          self.temp_points = []

      if not self.done:
          # update the param with current list of points
          param['roi_points'] = self.temp_points
          param['current_point'] = self.current

  def _handle_roi_setup(self, frame, window_name, params) -> None:
    """
    Handle the setup of the region of interest.

    Parameters
    ----------
    frame : np.ndarray
      The frame to display.
    window_name : str
      The name of the window.
    params : dict
      A dictionary containing the ROI points and the current point.
    """
    while not self.done:
      if len(params['roi_points']) > 0:
        # If the user has selected points, display the ROI points and the
        # current point
        updated_frame = frame.copy()
        cv2.polylines(
            updated_frame, [
                np.array(
                    params['roi_points'])], False, (0, 255, 0), 2)
        cv2.line(updated_frame, params['roi_points']
                 [-1], params['current_point'], (0, 255, 0), 2)
        cv2.imshow(window_name, updated_frame)
      else:
        cv2.imshow(window_name, frame)
      if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(
              window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

  def get_roi(self, frame: np.ndarray,
              window_name: str = 'Define Region of Interest') -> np.ndarray:
    """
    Set the region of interest in a frame.

    Intialise the region of interest by selecting the points on the frame using the mouse.
    The user can select the points by clicking on the frame to define the region of interest.
    The user can also clear the points by clicking the right mouse button and finish selecting
    the points by clicking the middle mouse button. The ROI points are saved to a numpy file
    for future use.


    Parameters
    ----------
    frame : np.ndarray
      The frame to select the region of interest.
    window_name : str, optional
      The name of the window, by default 'Define Region of Interest'.

    Returns
    -------
    np.ndarray
      The region of interest points.
    """
    cv2.namedWindow(window_name)

    if len(self.roi_points) > 0:
      # If ROI points are provided, display them on the frame
      cv2.polylines(frame, [np.array(self.roi_points)], False, (0, 255, 0), 2)

    params = {'roi_points': [], 'current_point': (0, 0)}
    cv2.setMouseCallback(window_name, self.handle_mouse_events, params)
    self._handle_roi_setup(frame, window_name, params)
    cv2.destroyWindow(window_name)
    
    return np.array(self.roi_points, dtype=np.int32)
